#!/usr/bin/env python3
"""
Speaker Separation using SepFormer
Separates two speakers (you vs singer) using a model trained for speaker separation
"""

import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import os
import torch


def load_audio(path: str, target_sr: int = 8000):
    """Load audio. SepFormer works at 8kHz."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Estimate delay using cross-correlation."""
    print("[2/5] Estimating delay...")
    chunk = min(sr * 2, len(mic))
    corr = signal.correlate(mic[:chunk], ref[:chunk], mode='same')
    delay = np.argmax(corr) - chunk // 2
    print(f"  Delay: {delay} samples ({delay*1000/sr:.1f}ms)")
    return delay


def align_signals(mic: np.ndarray, ref: np.ndarray, delay: int):
    """Align reference to mic."""
    if delay > 0:
        ref = np.concatenate([np.zeros(delay, dtype=np.float32), ref])[:len(mic)]
    elif delay < 0:
        ref = ref[-delay:]
    if len(ref) < len(mic):
        ref = np.concatenate([ref, np.zeros(len(mic) - len(ref), dtype=np.float32)])
    return ref[:len(mic)].astype(np.float32)


def run_demucs_vocals(audio: np.ndarray, orig_sr: int):
    """Extract vocals using Demucs first."""
    print("  Running Demucs vocal extraction...")
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = get_model('htdemucs')
    model.to(device)
    model.eval()
    
    # Demucs needs 44100
    audio_44k = signal.resample_poly(audio, 44100, orig_sr).astype(np.float32)
    
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    
    # Resample back
    vocals = signal.resample_poly(vocals, orig_sr, 44100).astype(np.float32)
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    return vocals.astype(np.float32)


def run_sepformer(audio: np.ndarray, sr: int):
    """Run SepFormer to separate two speakers."""
    print("[4/5] Running SepFormer speaker separation...")
    from speechbrain.inference.separation import SepformerSeparation
    
    # Load pre-trained model
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="pretrained_models/sepformer-wsj02mix"
    )
    
    # SepFormer expects 8kHz
    if sr != 8000:
        audio_8k = signal.resample_poly(audio, 8000, sr).astype(np.float32)
    else:
        audio_8k = audio
    
    # Prepare tensor
    audio_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)
    
    # Separate
    with torch.no_grad():
        est_sources = model.separate_batch(audio_tensor)
    
    # est_sources shape: [batch, time, num_speakers]
    speaker1 = est_sources[0, :, 0].numpy()
    speaker2 = est_sources[0, :, 1].numpy()
    
    # Resample back
    if sr != 8000:
        speaker1 = signal.resample_poly(speaker1, sr, 8000).astype(np.float32)
        speaker2 = signal.resample_poly(speaker2, sr, 8000).astype(np.float32)
    
    return speaker1, speaker2


def identify_your_voice(speaker1: np.ndarray, speaker2: np.ndarray, 
                        ref_vocals: np.ndarray, sr: int):
    """
    Identify which separated speaker is YOU (not in reference) vs SINGER (in reference).
    The singer's voice should correlate more with the reference vocals.
    """
    print("[5/5] Identifying your voice...")
    
    # Trim to same length
    min_len = min(len(speaker1), len(speaker2), len(ref_vocals))
    s1 = speaker1[:min_len]
    s2 = speaker2[:min_len]
    ref = ref_vocals[:min_len]
    
    # Compute correlation with reference
    corr1 = np.abs(np.corrcoef(s1, ref)[0, 1])
    corr2 = np.abs(np.corrcoef(s2, ref)[0, 1])
    
    print(f"  Speaker 1 correlation with singer: {corr1:.3f}")
    print(f"  Speaker 2 correlation with singer: {corr2:.3f}")
    
    # The one with LOWER correlation to reference is YOU
    if corr1 < corr2:
        print("  -> Speaker 1 is YOU")
        return speaker1
    else:
        print("  -> Speaker 2 is YOU")
        return speaker2


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80):
    """Remove low frequency rumble."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the SepFormer-based speaker separation pipeline."""
    print("=" * 60)
    print("SepFormer Speaker Separation")
    print("=" * 60)
    
    # Step 1: Load at 44.1kHz for Demucs
    print("[1/5] Loading audio...")
    mic_full, _ = sf.read(mic_path, dtype='float32')
    ref_full, ref_sr = sf.read(ref_path, dtype='float32')
    
    if mic_full.ndim > 1:
        mic_full = mic_full.mean(axis=1)
    if ref_full.ndim > 1:
        ref_full = ref_full.mean(axis=1)
    
    # Use 16kHz as working sample rate (good balance)
    work_sr = 16000
    mic = signal.resample_poly(mic_full, work_sr, ref_sr).astype(np.float32)
    ref = signal.resample_poly(ref_full, work_sr, ref_sr).astype(np.float32)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    mic = mic / (np.abs(mic).max() + 1e-8) * 0.95
    ref = ref / (np.abs(ref).max() + 1e-8) * 0.95
    
    print(f"  Loaded {len(mic)/work_sr:.1f}s at {work_sr}Hz")
    
    # Step 2: Align
    delay = estimate_delay(mic, ref, work_sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 3: Extract vocals from both using Demucs
    print("[3/5] Extracting vocals with Demucs...")
    mic_vocals = run_demucs_vocals(mic, work_sr)
    ref_vocals = run_demucs_vocals(ref_aligned, work_sr)
    
    # Step 4: Run SepFormer on mic vocals to separate you from singer
    speaker1, speaker2 = run_sepformer(mic_vocals, work_sr)
    
    # Step 5: Identify which speaker is you
    your_voice = identify_your_voice(speaker1, speaker2, ref_vocals, work_sr)
    
    # Trim to original length
    your_voice = your_voice[:len(mic)]
    if len(your_voice) < len(mic):
        your_voice = np.concatenate([your_voice, np.zeros(len(mic) - len(your_voice))])
    
    # Final processing
    final = highpass_filter(your_voice, work_sr, cutoff=80)
    
    # Normalize
    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, work_sr)
    print(f"\n✓ Saved: {output_path} ({work_sr}Hz)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)



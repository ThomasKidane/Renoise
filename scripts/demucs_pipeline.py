#!/usr/bin/env python3
"""
High-Quality Offline Echo Removal with Demucs
Uses Demucs for vocal extraction + spectral cleanup
"""

import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import os
import torch


def load_audio(path: str, target_sr: int = 44100):
    """Load audio at target sample rate."""
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


def run_demucs(audio: np.ndarray, sr: int):
    """Run Demucs to extract vocals."""
    print("[3/5] Running Demucs vocal extraction...")
    
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    # Get device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"  Using device: {device}")
    
    # Load model
    model = get_model('htdemucs')
    model.to(device)
    model.eval()
    
    # Demucs expects 44100 Hz
    if sr != 44100:
        audio_44k = signal.resample_poly(audio, 44100, sr).astype(np.float32)
    else:
        audio_44k = audio
    
    # Prepare tensor: [batch, channels, samples]
    # Demucs expects stereo, so duplicate mono
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
    audio_tensor = audio_tensor.repeat(1, 2, 1)  # [1, 2, samples]
    audio_tensor = audio_tensor.to(device)
    
    # Apply model
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    # sources shape: [batch, 4, channels, samples]
    # Sources: drums, bass, other, vocals (index 3)
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    
    # Resample back if needed
    if sr != 44100:
        vocals = signal.resample_poly(vocals, sr, 44100).astype(np.float32)
    
    # Trim to original length
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    print(f"  Demucs extraction complete")
    return vocals.astype(np.float32)


def spectral_cleanup(audio: np.ndarray, ref: np.ndarray, sr: int):
    """Light spectral cleanup to remove residual reference bleed."""
    print("[4/5] Spectral cleanup...")
    
    n_fft = 2048
    hop = 512
    
    _, _, audio_stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_frames = min(audio_stft.shape[1], ref_stft.shape[1])
    audio_stft = audio_stft[:, :min_frames]
    ref_stft = ref_stft[:, :min_frames]
    
    audio_mag = np.abs(audio_stft)
    ref_mag = np.abs(ref_stft)
    audio_phase = np.angle(audio_stft)
    
    # Gentle suppression where reference is strong
    eps = 1e-8
    ratio = audio_mag / (ref_mag + eps)
    
    # Soft mask - only suppress where ref clearly dominates
    threshold = 1.0
    mask = np.clip(ratio / threshold, 0.3, 1.0)
    
    clean_stft = audio_mag * mask * np.exp(1j * audio_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80):
    """Remove low frequency rumble."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the Demucs-based pipeline."""
    print("=" * 60)
    print("High-Quality Echo Removal with Demucs")
    print("=" * 60)
    
    # Step 1: Load at 44.1kHz
    print("[1/5] Loading audio...")
    mic, sr = load_audio(mic_path, 44100)
    ref, _ = load_audio(ref_path, 44100)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s at {sr}Hz")
    
    # Step 2: Align
    delay = estimate_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 3: Run Demucs
    vocals = run_demucs(mic, sr)
    
    # Step 4: Light spectral cleanup
    cleaned = spectral_cleanup(vocals, ref_aligned, sr)
    
    # Step 5: Final processing
    print("[5/5] Final processing...")
    final = highpass_filter(cleaned, sr, cutoff=80)
    
    # Normalize
    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, sr)
    print(f"\n✓ Saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)



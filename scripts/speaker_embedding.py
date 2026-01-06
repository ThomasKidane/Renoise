#!/usr/bin/env python3
"""
Speaker Separation using Voice Embeddings
Uses Resemblyzer to create speaker embeddings and separate based on similarity
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import argparse
import os
import torch


def load_audio(path: str, target_sr: int = 16000):
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
    print("[2/7] Estimating delay...")
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


def run_demucs(audio: np.ndarray, orig_sr: int):
    """Run Demucs to extract vocals."""
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


def get_speaker_embedding(audio: np.ndarray, sr: int):
    """Get speaker embedding using Resemblyzer."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    
    encoder = VoiceEncoder()
    
    # Resemblyzer expects 16kHz
    if sr != 16000:
        audio_16k = signal.resample_poly(audio, 16000, sr).astype(np.float32)
    else:
        audio_16k = audio
    
    # Preprocess
    wav = preprocess_wav(audio_16k, source_sr=16000)
    
    # Get embedding
    embed = encoder.embed_utterance(wav)
    
    return embed


def get_frame_embeddings(audio: np.ndarray, sr: int, frame_sec: float = 1.0, hop_sec: float = 0.25):
    """Get speaker embeddings for each frame."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    
    encoder = VoiceEncoder()
    
    if sr != 16000:
        audio_16k = signal.resample_poly(audio, 16000, sr).astype(np.float32)
        sr_work = 16000
    else:
        audio_16k = audio
        sr_work = sr
    
    frame_len = int(frame_sec * sr_work)
    hop_len = int(hop_sec * sr_work)
    
    n_frames = max(1, (len(audio_16k) - frame_len) // hop_len + 1)
    embeddings = []
    
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        if end > len(audio_16k):
            break
        
        frame = audio_16k[start:end]
        
        # Skip silent frames
        if np.abs(frame).max() < 0.01:
            embeddings.append(None)
            continue
        
        try:
            wav = preprocess_wav(frame, source_sr=16000)
            if len(wav) > 0:
                embed = encoder.embed_utterance(wav)
                embeddings.append(embed)
            else:
                embeddings.append(None)
        except:
            embeddings.append(None)
    
    return embeddings, hop_len, sr_work


def speaker_separation_embedding(mic_vocals: np.ndarray, ref_vocals: np.ndarray, sr: int):
    """
    Separate speakers using voice embeddings.
    Frames similar to singer embedding -> suppress
    Frames different from singer embedding -> keep
    """
    print("[5/7] Computing speaker embeddings...")
    
    # Get singer's embedding from reference
    singer_embed = get_speaker_embedding(ref_vocals, sr)
    print(f"  Got singer embedding")
    
    # Get frame-by-frame embeddings from mic
    print("  Computing frame embeddings...")
    frame_embeds, hop_len, sr_work = get_frame_embeddings(mic_vocals, sr, frame_sec=0.5, hop_sec=0.1)
    
    # Compute similarity to singer for each frame
    similarities = []
    for embed in frame_embeds:
        if embed is None:
            similarities.append(0.5)  # Neutral for silent frames
        else:
            # Cosine similarity
            sim = np.dot(embed, singer_embed) / (np.linalg.norm(embed) * np.linalg.norm(singer_embed) + 1e-8)
            similarities.append(sim)
    
    similarities = np.array(similarities)
    print(f"  Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
    
    # Smooth similarities
    similarities = gaussian_filter1d(similarities, sigma=2)
    
    # Create time-domain mask
    # High similarity to singer -> low gain (suppress)
    # Low similarity to singer -> high gain (keep)
    
    # Normalize similarities to [0, 1]
    sim_min, sim_max = similarities.min(), similarities.max()
    if sim_max > sim_min:
        similarities_norm = (similarities - sim_min) / (sim_max - sim_min)
    else:
        similarities_norm = np.ones_like(similarities) * 0.5
    
    # Invert: high similarity = low mask
    mask_frames = 1.0 - similarities_norm
    
    # Apply threshold to be more aggressive
    threshold = 0.4
    mask_frames = np.where(mask_frames > threshold, mask_frames, mask_frames * 0.3)
    
    # Interpolate mask to sample level
    n_samples = len(mic_vocals)
    frame_times = np.arange(len(mask_frames)) * hop_len / sr_work
    sample_times = np.arange(n_samples) / sr
    
    mask_samples = np.interp(sample_times, frame_times, mask_frames)
    
    # Smooth the sample-level mask
    mask_samples = gaussian_filter1d(mask_samples, sigma=int(sr * 0.02))
    
    # Apply mask
    output = mic_vocals * mask_samples
    
    return output.astype(np.float32), mask_samples


def spectral_cleanup(audio: np.ndarray, ref: np.ndarray, sr: int):
    """Additional spectral cleanup."""
    print("[6/7] Spectral cleanup...")
    
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
    
    # Suppress where reference is strong
    eps = 1e-8
    ratio = audio_mag / (ref_mag + eps)
    mask = np.clip(ratio / 2.0, 0.2, 1.0)
    
    clean_stft = audio_mag * mask * np.exp(1j * audio_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80):
    """Remove low frequency rumble."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the embedding-based speaker separation pipeline."""
    print("=" * 60)
    print("Speaker Separation using Voice Embeddings")
    print("=" * 60)
    
    # Step 1: Load at 16kHz (Resemblyzer native rate)
    print("[1/7] Loading audio...")
    mic, sr = load_audio(mic_path, 16000)
    ref, _ = load_audio(ref_path, 16000)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s at {sr}Hz")
    
    # Step 2: Align
    delay = estimate_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 3: Extract vocals from mic
    print("[3/7] Extracting vocals from mic (Demucs)...")
    mic_vocals = run_demucs(mic, sr)
    
    # Step 4: Extract vocals from reference
    print("[4/7] Extracting vocals from reference (Demucs)...")
    ref_vocals = run_demucs(ref_aligned, sr)
    
    # Step 5: Speaker separation using embeddings
    separated, mask = speaker_separation_embedding(mic_vocals, ref_vocals, sr)
    
    # Step 6: Spectral cleanup
    cleaned = spectral_cleanup(separated, ref_vocals, sr)
    
    # Step 7: Final processing
    print("[7/7] Final processing...")
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



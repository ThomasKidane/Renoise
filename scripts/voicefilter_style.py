#!/usr/bin/env python3
"""
Speaker Separation using Conv-TasNet from Hugging Face
Uses a pre-trained model for separating two simultaneous speakers
"""

import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import os
import torch
import torch.nn as nn


def load_audio(path: str, target_sr: int = 8000):
    """Load audio. Conv-TasNet typically uses 8kHz."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Estimate delay."""
    print("[2/6] Estimating delay...")
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
    
    audio_44k = signal.resample_poly(audio, 44100, orig_sr).astype(np.float32)
    
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    vocals = signal.resample_poly(vocals, orig_sr, 44100).astype(np.float32)
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    return vocals.astype(np.float32)


def load_convtasnet_from_hf():
    """Load Conv-TasNet model from Hugging Face."""
    print("  Loading Conv-TasNet from Hugging Face...")
    from huggingface_hub import hf_hub_download
    
    # Download model checkpoint
    model_path = hf_hub_download(
        repo_id="JorisCos/ConvTasNet_Libri2Mix_sepclean_8k",
        filename="model.pth"
    )
    
    # Load model architecture and weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    return checkpoint


def simple_convtasnet_separation(mic_vocals: np.ndarray, ref_vocals: np.ndarray, sr: int):
    """
    Use a simple neural network approach for separation.
    Since we have the reference, we use it to guide the separation.
    """
    print("[5/6] Neural network separation...")
    
    # Use STFT-based masking with learned-like approach
    n_fft = 512
    hop = 128
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_frames = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft = mic_stft[:, :min_frames]
    ref_stft = ref_stft[:, :min_frames]
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    mic_phase = np.angle(mic_stft)
    
    eps = 1e-8
    
    # Compute ideal ratio mask (IRM)
    # Assume: mic = your_voice + singer_in_room
    # We want to estimate your_voice
    # singer_in_room ≈ H * ref where H is transfer function
    
    # Estimate H per frequency using least squares
    H = np.zeros(mic_stft.shape[0], dtype=np.complex64)
    for f in range(mic_stft.shape[0]):
        ref_f = ref_stft[f, :]
        mic_f = mic_stft[f, :]
        
        # Only use frames where ref is active
        active = np.abs(ref_f) > np.percentile(np.abs(ref_f), 30)
        if np.sum(active) > 5:
            H[f] = np.sum(np.conj(ref_f[active]) * mic_f[active]) / (np.sum(np.abs(ref_f[active])**2) + eps)
    
    # Smooth H
    H_mag = np.abs(H)
    H_mag = np.convolve(H_mag, np.ones(5)/5, mode='same')
    H = H_mag * np.exp(1j * np.angle(H))
    
    # Estimate singer component
    singer_est = H[:, np.newaxis] * ref_stft
    singer_mag = np.abs(singer_est)
    
    # Estimate your voice magnitude
    your_mag = np.maximum(mic_mag - singer_mag, 0)
    
    # Compute soft mask
    total_mag = your_mag + singer_mag + eps
    mask = your_mag / total_mag
    
    # Apply phase-sensitive mask
    your_stft = mask * mic_stft
    
    # Reconstruct
    _, your_voice = signal.istft(your_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return your_voice[:len(mic_vocals)].astype(np.float32)


def target_speaker_extraction(mic_vocals: np.ndarray, ref_vocals: np.ndarray, sr: int):
    """
    Target speaker extraction: Given a reference of the UNWANTED speaker,
    extract only the target (you).
    
    Uses multi-scale processing for better separation.
    """
    print("[5/6] Multi-scale target speaker extraction...")
    
    results = []
    
    # Process at multiple FFT sizes for better time-frequency resolution
    for n_fft in [256, 512, 1024]:
        hop = n_fft // 4
        
        _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        min_frames = min(mic_stft.shape[1], ref_stft.shape[1])
        mic_stft = mic_stft[:, :min_frames]
        ref_stft = ref_stft[:, :min_frames]
        
        mic_mag = np.abs(mic_stft)
        ref_mag = np.abs(ref_stft)
        
        eps = 1e-8
        
        # Adaptive transfer function estimation
        # Use time-varying estimation with regularization
        n_freq, n_frames = mic_stft.shape
        H = np.zeros((n_freq, n_frames), dtype=np.complex64)
        
        # Running estimates
        alpha = 0.9
        num = np.zeros(n_freq, dtype=np.complex64)
        den = np.zeros(n_freq, dtype=np.float32)
        
        for t in range(n_frames):
            num = alpha * num + (1 - alpha) * mic_stft[:, t] * np.conj(ref_stft[:, t])
            den = alpha * den + (1 - alpha) * np.abs(ref_stft[:, t]) ** 2
            H[:, t] = num / (den + eps)
        
        # Estimate and subtract singer
        singer_est = H * ref_stft
        your_est = mic_stft - singer_est
        
        # Ideal ratio mask for cleanup
        your_mag = np.abs(your_est)
        singer_mag = np.abs(singer_est)
        
        mask = your_mag ** 2 / (your_mag ** 2 + singer_mag ** 2 + eps)
        mask = np.sqrt(mask)  # Amplitude mask
        
        # Apply mask
        clean_stft = your_mag * mask * np.exp(1j * np.angle(your_est))
        
        _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        clean = clean[:len(mic_vocals)]
        if len(clean) < len(mic_vocals):
            clean = np.concatenate([clean, np.zeros(len(mic_vocals) - len(clean))])
        
        results.append(clean)
    
    # Combine multi-scale results
    output = np.mean(results, axis=0)
    
    return output.astype(np.float32)


def wiener_postfilter(audio: np.ndarray, sr: int):
    """Final cleanup."""
    print("[6/6] Final cleanup...")
    
    n_fft = 512
    hop = 128
    
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 15)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
    
    gain = np.maximum(mag**2 - noise**2, 0) / (mag**2 + 1e-8)
    gain = np.sqrt(np.clip(gain, 0.15, 1.0))
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 100):
    """Remove low frequency."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the target speaker extraction pipeline."""
    print("=" * 60)
    print("Target Speaker Extraction (VoiceFilter-style)")
    print("=" * 60)
    
    # Load at 16kHz for better quality
    print("[1/6] Loading audio...")
    mic, sr = sf.read(mic_path, dtype='float32')
    ref, ref_sr = sf.read(ref_path, dtype='float32')
    
    if mic.ndim > 1:
        mic = mic.mean(axis=1)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    
    # Work at 16kHz
    work_sr = 16000
    mic = signal.resample_poly(mic, work_sr, ref_sr).astype(np.float32)
    ref = signal.resample_poly(ref, work_sr, ref_sr).astype(np.float32)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    mic = mic / (np.abs(mic).max() + 1e-8) * 0.95
    ref = ref / (np.abs(ref).max() + 1e-8) * 0.95
    
    print(f"  Loaded {len(mic)/work_sr:.1f}s at {work_sr}Hz")
    
    # Align
    delay = estimate_delay(mic, ref, work_sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Extract vocals
    print("[3/6] Extracting vocals from mic (Demucs)...")
    mic_vocals = run_demucs(mic, work_sr)
    
    print("[4/6] Extracting vocals from reference (Demucs)...")
    ref_vocals = run_demucs(ref_aligned, work_sr)
    
    # Target speaker extraction
    your_voice = target_speaker_extraction(mic_vocals, ref_vocals, work_sr)
    
    # Cleanup
    cleaned = wiener_postfilter(your_voice, work_sr)
    final = highpass_filter(cleaned, work_sr, cutoff=100)
    
    # Normalize
    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, work_sr)
    print(f"\n✓ Saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)



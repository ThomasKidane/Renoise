#!/usr/bin/env python3
"""
Offline Speaker Playback Removal - Memory Efficient Version
Processes audio in small chunks to avoid OOM
"""

import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import os

def load_audio(mic_path: str, ref_path: str, target_sr: int = 16000):
    """Load and downsample to 16kHz for efficiency."""
    print(f"[1/5] Loading audio...")
    
    mic, mic_sr = sf.read(mic_path, dtype='float32')
    ref, ref_sr = sf.read(ref_path, dtype='float32')
    
    # Mono
    if mic.ndim > 1:
        mic = mic.mean(axis=1)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    
    # Downsample to 16kHz
    if mic_sr != target_sr:
        mic = signal.resample_poly(mic, target_sr, mic_sr).astype(np.float32)
    if ref_sr != target_sr:
        ref = signal.resample_poly(ref, target_sr, ref_sr).astype(np.float32)
    
    # Same length
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    # Normalize
    mic = mic / (np.abs(mic).max() + 1e-8) * 0.9
    ref = ref / (np.abs(ref).max() + 1e-8) * 0.9
    
    print(f"  Loaded {len(mic)/target_sr:.1f}s at {target_sr}Hz")
    return mic, ref, target_sr


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Estimate delay using cross-correlation on first 1 second."""
    print(f"[2/5] Estimating delay...")
    
    chunk = min(sr, len(mic))  # 1 second max
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
    
    return ref[:len(mic)]


def process_stft_chunked(mic: np.ndarray, ref: np.ndarray, sr: int):
    """STFT-based echo removal with spectral gating - gentler settings."""
    print(f"[4/5] STFT spectral processing...")
    
    n_fft = 512
    hop = 128
    chunk_samples = sr * 5  # 5 second chunks
    output = np.zeros_like(mic)
    
    for start in range(0, len(mic), chunk_samples):
        end = min(start + chunk_samples + n_fft, len(mic))
        mic_chunk = mic[start:end]
        ref_chunk = ref[start:end]
        
        if len(mic_chunk) < n_fft:
            output[start:end] = mic_chunk
            continue
        
        # STFT
        _, _, mic_stft = signal.stft(mic_chunk, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, ref_stft = signal.stft(ref_chunk, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        mic_mag = np.abs(mic_stft)
        ref_mag = np.abs(ref_stft)
        mic_phase = np.angle(mic_stft)
        
        # Gentler spectral subtraction - only suppress where ref is dominant
        alpha = 1.0  # Subtraction factor (was 2.0)
        beta = 0.15   # Minimum gain floor (was 0.05)
        
        # Only subtract where reference is significant
        ref_threshold = np.percentile(ref_mag, 30)
        ref_mask = ref_mag > ref_threshold
        
        gain = np.ones_like(mic_mag)
        # Where ref is present, apply soft suppression
        suppression = np.clip(1.0 - alpha * ref_mag / (mic_mag + 1e-8), beta, 1.0)
        gain = np.where(ref_mask, suppression, gain)
        
        clean_stft = mic_mag * gain * np.exp(1j * mic_phase)
        _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        # Overlap-add
        out_end = min(start + len(clean), len(output))
        output[start:out_end] = clean[:out_end - start]
    
    print(f"  STFT processing complete")
    return output


def nlms_filter(mic: np.ndarray, ref: np.ndarray, sr: int, filter_len: int = 512, mu: float = 0.05):
    """NLMS adaptive filter for echo cancellation - gentler settings."""
    print(f"[3/5] NLMS adaptive filtering...")
    
    n = len(mic)
    w = np.zeros(filter_len, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)
    
    # Copy beginning where filter hasn't adapted yet
    output[:filter_len] = mic[:filter_len]
    
    for i in range(filter_len, n):
        x = ref[i-filter_len:i][::-1]
        y_hat = np.dot(w, x)
        e = mic[i] - y_hat
        
        # NLMS update with smaller step size to preserve voice
        norm = np.dot(x, x) + 1e-4
        w = w + mu * e * x / norm
        
        output[i] = e
    
    # Normalize to match input level
    input_rms = np.sqrt(np.mean(mic**2))
    output_rms = np.sqrt(np.mean(output**2)) + 1e-8
    output = output * (input_rms / output_rms) * 0.8
    
    print(f"  NLMS complete")
    return output


def wiener_postfilter(audio: np.ndarray, sr: int):
    """Simple Wiener postfilter for cleanup - gentler."""
    print(f"[5/5] Wiener postfilter...")
    
    n_fft = 512
    hop = 128
    
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from quietest 5% of frames (was 10%)
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 20)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.mean(mag[:, quiet_idx], axis=1, keepdims=True)
    
    # Gentler Wiener gain with higher floor
    gain = np.maximum(mag**2 - 0.5 * noise**2, 0) / (mag**2 + 1e-8)
    gain = np.sqrt(np.clip(gain, 0.3, 1.0))  # Higher floor (was 0.1)
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    clean = clean[:len(audio)]
    if len(clean) < len(audio):
        clean = np.concatenate([clean, np.zeros(len(audio) - len(clean))])
    
    print(f"  Postfilter complete")
    return clean.astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the memory-efficient pipeline."""
    print("=" * 50)
    print("Offline Echo Removal (Memory Efficient)")
    print("=" * 50)
    
    # Load at 16kHz
    mic, ref, sr = load_audio(mic_path, ref_path, target_sr=16000)
    
    # Align
    delay = estimate_delay(mic, ref, sr)
    ref = align_signals(mic, ref, delay)
    
    # NLMS first pass
    nlms_out = nlms_filter(mic, ref, sr)
    
    # STFT spectral processing
    stft_out = process_stft_chunked(nlms_out, ref, sr)
    
    # Wiener postfilter
    final = wiener_postfilter(stft_out, sr)
    
    # Normalize
    final = final / (np.abs(final).max() + 1e-8) * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, sr)
    print(f"\n✓ Saved: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)

#!/usr/bin/env python3
"""
High-Quality Offline Echo Removal Pipeline
Pure NumPy/SciPy implementation at 44.1kHz
No external deep learning dependencies
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft, rfft, irfft
import argparse
import os


def load_audio(path: str, target_sr: int = 44100):
    """Load audio at high sample rate."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def estimate_delay_gcc_phat(mic: np.ndarray, ref: np.ndarray, sr: int, max_delay_ms: float = 200):
    """Estimate delay using GCC-PHAT on first 2 seconds."""
    print("[2/7] Estimating delay (GCC-PHAT)...")
    
    chunk_len = min(sr * 2, len(mic))
    mic_chunk = mic[:chunk_len]
    ref_chunk = ref[:chunk_len]
    
    # FFT-based cross-correlation with phase transform
    n_fft = 2 ** int(np.ceil(np.log2(2 * chunk_len)))
    
    mic_fft = fft(mic_chunk, n_fft)
    ref_fft = fft(ref_chunk, n_fft)
    
    # GCC-PHAT
    cross = mic_fft * np.conj(ref_fft)
    cross_norm = cross / (np.abs(cross) + 1e-10)
    gcc = np.real(ifft(cross_norm))
    
    # Find peak in valid range
    max_samples = int(max_delay_ms * sr / 1000)
    gcc_valid = np.concatenate([gcc[-max_samples:], gcc[:max_samples]])
    peak_idx = np.argmax(np.abs(gcc_valid))
    delay = peak_idx - max_samples
    
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


def frequency_domain_aec(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Frequency-domain adaptive echo cancellation."""
    print("[3/7] Frequency-domain AEC...")
    
    n_fft = 2048
    hop = 512
    
    # STFT
    _, _, mic_stft = signal.stft(mic, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    n_freq, n_frames = mic_stft.shape
    
    # Adaptive filter per frequency bin
    # Use recursive least squares style estimation
    output_stft = np.zeros_like(mic_stft)
    
    # Filter coefficients (one per frequency)
    W = np.zeros(n_freq, dtype=np.complex64)
    
    # Forgetting factor
    lambda_f = 0.98
    P = np.ones(n_freq) * 100  # Inverse correlation estimate
    
    for t in range(n_frames):
        x = ref_stft[:, t]  # Reference
        d = mic_stft[:, t]  # Desired (mic)
        
        # Predicted echo
        y = W * x
        
        # Error
        e = d - y
        
        # RLS update
        x_conj = np.conj(x)
        k = P * x_conj / (lambda_f + P * np.abs(x)**2 + 1e-8)
        W = W + k * e
        P = (P - k * x * P) / lambda_f
        P = np.clip(P, 0.01, 1000)  # Stability
        
        output_stft[:, t] = e
    
    # ISTFT
    _, output = signal.istft(output_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    output = output[:len(mic)]
    if len(output) < len(mic):
        output = np.concatenate([output, np.zeros(len(mic) - len(output))])
    
    return output.astype(np.float32)


def spectral_subtraction(audio: np.ndarray, ref: np.ndarray, sr: int):
    """Multi-band spectral subtraction."""
    print("[4/7] Spectral subtraction...")
    
    n_fft = 2048
    hop = 512
    
    _, _, audio_stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    # Match shapes
    min_frames = min(audio_stft.shape[1], ref_stft.shape[1])
    audio_stft = audio_stft[:, :min_frames]
    ref_stft = ref_stft[:, :min_frames]
    
    audio_mag = np.abs(audio_stft)
    ref_mag = np.abs(ref_stft)
    audio_phase = np.angle(audio_stft)
    
    # Frequency-dependent subtraction factor
    # More aggressive at low frequencies where music is stronger
    n_freq = audio_mag.shape[0]
    freqs = np.linspace(0, sr/2, n_freq)
    alpha = 1.0 + 1.0 * np.exp(-freqs / 500)  # Higher at low freq
    alpha = alpha[:, np.newaxis]
    
    # Spectral subtraction with floor
    beta = 0.1  # Spectral floor
    clean_mag = np.maximum(audio_mag - alpha * ref_mag, beta * audio_mag)
    
    # Smooth across time to reduce musical noise
    smooth_kernel = np.array([0.25, 0.5, 0.25])
    for f in range(n_freq):
        clean_mag[f, :] = np.convolve(clean_mag[f, :], smooth_kernel, mode='same')
    
    clean_stft = clean_mag * np.exp(1j * audio_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def spectral_gate(audio: np.ndarray, ref: np.ndarray, sr: int):
    """Soft spectral gating based on SNR."""
    print("[5/7] Spectral gating...")
    
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
    
    # Compute local SNR
    eps = 1e-8
    snr = audio_mag / (ref_mag + eps)
    
    # Sigmoid mask - keep where audio dominates, suppress where ref dominates
    threshold = 2.0
    steepness = 1.5
    mask = 1.0 / (1.0 + np.exp(-steepness * (snr - threshold)))
    
    # Floor to preserve some signal
    mask = np.maximum(mask, 0.15)
    
    # Smooth mask
    smooth_kernel = np.array([0.2, 0.6, 0.2])
    for f in range(mask.shape[0]):
        mask[f, :] = np.convolve(mask[f, :], smooth_kernel, mode='same')
    
    clean_stft = audio_mag * mask * np.exp(1j * audio_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def wiener_filter(audio: np.ndarray, sr: int):
    """Wiener filter for residual noise."""
    print("[6/7] Wiener filter...")
    
    n_fft = 2048
    hop = 512
    
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from quietest 5% of frames
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 20)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
    
    # Wiener gain
    signal_var = np.maximum(mag**2 - noise**2, 0)
    gain = signal_var / (mag**2 + 1e-8)
    gain = np.sqrt(np.clip(gain, 0.2, 1.0))
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80):
    """Remove low frequency rumble."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the high-quality pipeline."""
    print("=" * 60)
    print("High-Quality Offline Echo Removal (44.1kHz)")
    print("=" * 60)
    
    # Step 1: Load
    print("[1/7] Loading audio at 44.1kHz...")
    mic, sr = load_audio(mic_path, 44100)
    ref, _ = load_audio(ref_path, 44100)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s")
    
    # Step 2: Align
    delay = estimate_delay_gcc_phat(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 3: Frequency-domain AEC
    aec_out = frequency_domain_aec(mic, ref_aligned, sr)
    
    # Step 4: Spectral subtraction
    sub_out = spectral_subtraction(aec_out, ref_aligned, sr)
    
    # Step 5: Spectral gating
    gated = spectral_gate(sub_out, ref_aligned, sr)
    
    # Step 6: Wiener filter
    wiener_out = wiener_filter(gated, sr)
    
    # Step 7: Final processing
    print("[7/7] Final processing...")
    final = highpass_filter(wiener_out, sr, cutoff=80)
    
    # Normalize
    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, sr)
    print(f"\n✓ Saved: {output_path} (44.1kHz)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)

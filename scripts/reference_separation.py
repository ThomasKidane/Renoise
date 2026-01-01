#!/usr/bin/env python3
"""
High-Quality Reference-Based Audio Separation

This script uses the reference signal (what's playing through speakers) to 
precisely remove it from the microphone recording. This is much better than
blind source separation when you have the exact reference.

Methods:
1. Adaptive NLMS Filter - learns the acoustic path and subtracts
2. Spectral Subtraction - frequency-domain removal
3. Phase-aligned subtraction - time-domain with cross-correlation alignment

Usage:
    python reference_separation.py --mic <mic.wav> --ref <ref.wav> --output <output.wav>
"""

import argparse
import os
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

def load_wav(path):
    """Load WAV file and convert to float32 normalized"""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass  # Already float
    else:
        data = data.astype(np.float32)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    return sr, data

def save_wav(path, sr, data):
    """Save as 32-bit float WAV"""
    # Clip to valid range
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(path, sr, data.astype(np.float32))

def align_signals(mic, ref, sr):
    """Align reference to microphone using cross-correlation"""
    # Use a chunk for faster correlation
    chunk_size = min(sr * 5, len(mic), len(ref))  # 5 seconds max
    
    mic_chunk = mic[:chunk_size]
    ref_chunk = ref[:chunk_size]
    
    # Cross-correlation to find delay
    correlation = signal.correlate(mic_chunk, ref_chunk, mode='full')
    lag = np.argmax(np.abs(correlation)) - len(ref_chunk) + 1
    
    print(f"  Detected delay: {lag} samples ({lag/sr*1000:.1f} ms)")
    
    # Align the signals
    if lag > 0:
        # Reference is ahead, shift it back
        ref_aligned = np.pad(ref, (lag, 0))[:len(mic)]
    elif lag < 0:
        # Reference is behind, shift it forward
        ref_aligned = ref[-lag:]
        if len(ref_aligned) < len(mic):
            ref_aligned = np.pad(ref_aligned, (0, len(mic) - len(ref_aligned)))
        else:
            ref_aligned = ref_aligned[:len(mic)]
    else:
        ref_aligned = ref[:len(mic)] if len(ref) >= len(mic) else np.pad(ref, (0, len(mic) - len(ref)))
    
    return ref_aligned

def nlms_filter(mic, ref, filter_length=4096, mu=0.1):
    """
    Normalized Least Mean Squares adaptive filter
    This learns the acoustic transfer function and subtracts the estimated echo
    """
    n_samples = len(mic)
    
    # Ensure same length
    if len(ref) < n_samples:
        ref = np.pad(ref, (0, n_samples - len(ref)))
    else:
        ref = ref[:n_samples]
    
    # Initialize
    w = np.zeros(filter_length)  # Filter coefficients
    output = np.zeros(n_samples)
    
    # Process
    for n in range(filter_length, n_samples):
        # Get reference buffer
        x = ref[n-filter_length:n][::-1]  # Reversed for convolution
        
        # Estimate echo
        y_hat = np.dot(w, x)
        
        # Error (desired output)
        e = mic[n] - y_hat
        output[n] = e
        
        # Update filter (NLMS)
        norm = np.dot(x, x) + 1e-10
        w = w + (mu / norm) * e * x
    
    return output

def spectral_subtraction(mic, ref, sr, alpha=1.0, beta=0.01):
    """
    Spectral subtraction in frequency domain
    Removes reference spectrum from microphone spectrum
    """
    # STFT parameters
    n_fft = 2048
    hop_length = n_fft // 4
    
    # Ensure same length
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    # Compute STFTs
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    _, _, ref_stft = signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Magnitude and phase
    mic_mag = np.abs(mic_stft)
    mic_phase = np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Estimate scaling factor per frame
    # This accounts for volume differences
    scale_factors = []
    for i in range(mic_mag.shape[1]):
        ref_energy = np.sum(ref_mag[:, i] ** 2) + 1e-10
        cross_energy = np.sum(mic_mag[:, i] * ref_mag[:, i])
        scale = cross_energy / ref_energy
        scale_factors.append(max(0, min(scale, 2.0)))  # Clamp
    
    scale_factors = np.array(scale_factors)
    # Smooth the scale factors
    scale_factors = signal.medfilt(scale_factors, kernel_size=5)
    
    # Subtract
    output_mag = np.zeros_like(mic_mag)
    for i in range(mic_mag.shape[1]):
        subtracted = mic_mag[:, i] - alpha * scale_factors[i] * ref_mag[:, i]
        # Spectral floor to avoid musical noise
        output_mag[:, i] = np.maximum(subtracted, beta * mic_mag[:, i])
    
    # Reconstruct
    output_stft = output_mag * np.exp(1j * mic_phase)
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    return output

def wiener_filter(mic, ref, sr):
    """
    Wiener filter for optimal noise reduction
    Uses reference as noise estimate
    """
    n_fft = 2048
    hop_length = n_fft // 4
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    # Compute STFTs
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    _, _, ref_stft = signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    mic_power = np.abs(mic_stft) ** 2
    ref_power = np.abs(ref_stft) ** 2
    
    # Estimate noise power (scaled reference)
    # Use median to estimate typical noise level
    noise_scale = np.median(mic_power) / (np.median(ref_power) + 1e-10)
    noise_power = noise_scale * ref_power
    
    # Wiener filter gain
    gain = np.maximum(mic_power - noise_power, 0) / (mic_power + 1e-10)
    gain = np.sqrt(gain)  # Amplitude domain
    
    # Smooth gain to reduce musical noise
    for i in range(gain.shape[0]):
        gain[i, :] = signal.medfilt(gain[i, :], kernel_size=3)
    
    # Apply gain
    output_stft = gain * mic_stft
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    return output

def hybrid_separation(mic, ref, sr):
    """
    Hybrid approach combining multiple methods for best results
    """
    print("  Step 1: Aligning signals...")
    ref_aligned = align_signals(mic, ref, sr)
    
    print("  Step 2: Applying NLMS adaptive filter...")
    nlms_output = nlms_filter(mic, ref_aligned, filter_length=8192, mu=0.05)
    
    print("  Step 3: Applying spectral subtraction...")
    spectral_output = spectral_subtraction(nlms_output, ref_aligned, sr, alpha=0.8, beta=0.02)
    
    print("  Step 4: Applying Wiener filter for cleanup...")
    final_output = wiener_filter(spectral_output, ref_aligned, sr)
    
    # Normalize output
    max_val = np.max(np.abs(final_output))
    if max_val > 0:
        final_output = final_output / max_val * 0.9
    
    return final_output

def main():
    parser = argparse.ArgumentParser(description='Reference-based audio separation')
    parser.add_argument('--mic', '-m', required=True, help='Microphone recording (with voice + music)')
    parser.add_argument('--ref', '-r', required=True, help='Reference signal (music playing through speakers)')
    parser.add_argument('--output', '-o', required=True, help='Output file (clean voice)')
    parser.add_argument('--method', default='hybrid', 
                        choices=['nlms', 'spectral', 'wiener', 'hybrid'],
                        help='Separation method (default: hybrid)')
    args = parser.parse_args()
    
    if not os.path.exists(args.mic):
        print(f"Error: Microphone file not found: {args.mic}")
        sys.exit(1)
    if not os.path.exists(args.ref):
        print(f"Error: Reference file not found: {args.ref}")
        sys.exit(1)
    
    print(f"Loading audio files...")
    sr_mic, mic = load_wav(args.mic)
    sr_ref, ref = load_wav(args.ref)
    
    print(f"  Microphone: {len(mic)/sr_mic:.1f}s @ {sr_mic}Hz")
    print(f"  Reference: {len(ref)/sr_ref:.1f}s @ {sr_ref}Hz")
    
    # Resample if needed
    if sr_ref != sr_mic:
        print(f"  Resampling reference from {sr_ref}Hz to {sr_mic}Hz...")
        ref = signal.resample(ref, int(len(ref) * sr_mic / sr_ref))
    
    print(f"\nProcessing with {args.method} method...")
    
    if args.method == 'nlms':
        ref_aligned = align_signals(mic, ref, sr_mic)
        output = nlms_filter(mic, ref_aligned, filter_length=8192, mu=0.1)
    elif args.method == 'spectral':
        ref_aligned = align_signals(mic, ref, sr_mic)
        output = spectral_subtraction(mic, ref_aligned, sr_mic)
    elif args.method == 'wiener':
        ref_aligned = align_signals(mic, ref, sr_mic)
        output = wiener_filter(mic, ref_aligned, sr_mic)
    else:  # hybrid
        output = hybrid_separation(mic, ref, sr_mic)
    
    print(f"\nSaving output to: {args.output}")
    save_wav(args.output, sr_mic, output)
    
    print("Done!")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Reference-Based Echo Cancellation v2

Fixed version that properly subtracts the reference signal from the microphone.
Uses careful gain estimation to avoid amplifying the music.

Usage:
    python reference_separation_v2.py --mic <mic.wav> --ref <ref.wav> --output <output.wav>
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
        pass
    else:
        data = data.astype(np.float32)
    
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    return sr, data

def save_wav(path, sr, data):
    """Save as 32-bit float WAV"""
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(path, sr, data.astype(np.float32))

def find_delay(mic, ref, sr, max_delay_ms=500):
    """Find the delay between mic and reference using cross-correlation"""
    max_delay_samples = int(max_delay_ms * sr / 1000)
    
    # Use a portion of the signal for speed
    chunk_len = min(sr * 3, len(mic), len(ref))  # 3 seconds
    
    mic_chunk = mic[:chunk_len]
    ref_chunk = ref[:chunk_len]
    
    # Normalize for better correlation
    mic_chunk = mic_chunk / (np.max(np.abs(mic_chunk)) + 1e-10)
    ref_chunk = ref_chunk / (np.max(np.abs(ref_chunk)) + 1e-10)
    
    # Cross-correlation
    corr = signal.correlate(mic_chunk, ref_chunk, mode='full')
    
    # Find peak within reasonable delay range
    mid = len(ref_chunk) - 1
    search_start = max(0, mid - max_delay_samples)
    search_end = min(len(corr), mid + max_delay_samples)
    
    peak_idx = search_start + np.argmax(np.abs(corr[search_start:search_end]))
    delay = peak_idx - mid
    
    return delay

def align_and_scale_reference(mic, ref, delay, sr):
    """Align reference to mic and estimate proper scaling"""
    n = len(mic)
    
    # Shift reference
    if delay > 0:
        # Mic is ahead of reference (reference delayed)
        ref_aligned = np.zeros(n)
        if delay < n:
            copy_len = min(len(ref), n - delay)
            ref_aligned[delay:delay + copy_len] = ref[:copy_len]
    elif delay < 0:
        # Reference is ahead of mic
        ref_aligned = np.zeros(n)
        start = -delay
        if start < len(ref):
            copy_len = min(len(ref) - start, n)
            ref_aligned[:copy_len] = ref[start:start + copy_len]
    else:
        ref_aligned = ref[:n] if len(ref) >= n else np.pad(ref, (0, n - len(ref)))
    
    # Estimate scaling factor using least squares
    # We want to find 'a' such that mic ≈ voice + a*ref_aligned
    # In regions where voice is quiet, mic ≈ a*ref_aligned
    
    # Use RMS-based scaling
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref_aligned ** 2))
    
    if ref_rms > 1e-10:
        # Start with RMS ratio
        scale = mic_rms / ref_rms
        
        # Refine using correlation in high-energy regions
        # (where music is likely dominant)
        threshold = 0.3 * np.max(np.abs(ref_aligned))
        high_energy_mask = np.abs(ref_aligned) > threshold
        
        if np.sum(high_energy_mask) > 100:
            mic_high = mic[high_energy_mask]
            ref_high = ref_aligned[high_energy_mask]
            
            # Least squares: scale = (mic · ref) / (ref · ref)
            scale = np.dot(mic_high, ref_high) / (np.dot(ref_high, ref_high) + 1e-10)
            scale = max(0.1, min(scale, 3.0))  # Clamp to reasonable range
    else:
        scale = 1.0
    
    return ref_aligned, scale

def simple_subtraction(mic, ref_aligned, scale, smoothing=0.95):
    """Simple time-domain subtraction with smoothing"""
    output = mic - scale * ref_aligned
    return output

def frequency_domain_subtraction(mic, ref_aligned, sr, scale, over_subtraction=1.0, floor=0.02):
    """
    Frequency-domain subtraction with proper phase handling
    """
    n_fft = 2048
    hop = n_fft // 4
    window = signal.windows.hann(n_fft)
    
    # Pad signals
    pad_len = n_fft - (len(mic) % hop)
    mic_padded = np.pad(mic, (0, pad_len))
    ref_padded = np.pad(ref_aligned, (0, pad_len))
    
    # STFT
    _, _, mic_stft = signal.stft(mic_padded, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_padded, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    
    # Magnitude and phase
    mic_mag = np.abs(mic_stft)
    mic_phase = np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Subtract scaled reference magnitude
    output_mag = mic_mag - over_subtraction * scale * ref_mag
    
    # Apply floor to prevent negative values and musical noise
    output_mag = np.maximum(output_mag, floor * mic_mag)
    
    # Reconstruct with original phase
    output_stft = output_mag * np.exp(1j * mic_phase)
    
    # ISTFT
    _, output = signal.istft(output_stft, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def adaptive_subtraction(mic, ref_aligned, sr, scale):
    """
    Adaptive subtraction that varies the subtraction amount based on 
    how well the reference matches the mic at each moment
    """
    n_fft = 2048
    hop = n_fft // 4
    window = signal.windows.hann(n_fft)
    
    pad_len = n_fft - (len(mic) % hop)
    mic_padded = np.pad(mic, (0, pad_len))
    ref_padded = np.pad(ref_aligned, (0, pad_len))
    
    _, _, mic_stft = signal.stft(mic_padded, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_padded, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    mic_phase = np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    output_mag = np.zeros_like(mic_mag)
    
    for t in range(mic_mag.shape[1]):
        mic_frame = mic_mag[:, t]
        ref_frame = ref_mag[:, t]
        
        # Estimate local scale for this frame
        ref_energy = np.sum(ref_frame ** 2) + 1e-10
        
        if ref_energy > 1e-6:  # Reference has energy
            # Wiener-like gain estimation
            # Assume mic = voice + scale * ref
            # We want to estimate how much of mic is from ref
            
            cross_energy = np.sum(mic_frame * ref_frame)
            local_scale = cross_energy / ref_energy
            local_scale = max(0, min(local_scale, scale * 1.5))  # Clamp
            
            # Subtract
            subtracted = mic_frame - local_scale * ref_frame
            
            # Floor
            output_mag[:, t] = np.maximum(subtracted, 0.01 * mic_frame)
        else:
            # No reference energy, keep mic as is
            output_mag[:, t] = mic_frame
    
    # Smooth across time to reduce artifacts
    for f in range(output_mag.shape[0]):
        output_mag[f, :] = signal.medfilt(output_mag[f, :], kernel_size=3)
    
    output_stft = output_mag * np.exp(1j * mic_phase)
    _, output = signal.istft(output_stft, sr, window=window, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def main():
    parser = argparse.ArgumentParser(description='Reference-based echo cancellation v2')
    parser.add_argument('--mic', '-m', required=True, help='Microphone recording')
    parser.add_argument('--ref', '-r', required=True, help='Reference signal')
    parser.add_argument('--output', '-o', required=True, help='Output file')
    parser.add_argument('--method', default='adaptive',
                        choices=['simple', 'spectral', 'adaptive'],
                        help='Method (default: adaptive)')
    args = parser.parse_args()
    
    print("Loading audio...")
    sr_mic, mic = load_wav(args.mic)
    sr_ref, ref = load_wav(args.ref)
    
    print(f"  Mic: {len(mic)/sr_mic:.1f}s @ {sr_mic}Hz")
    print(f"  Ref: {len(ref)/sr_ref:.1f}s @ {sr_ref}Hz")
    
    # Resample if needed
    if sr_ref != sr_mic:
        print(f"  Resampling reference...")
        ref = signal.resample(ref, int(len(ref) * sr_mic / sr_ref))
    
    # Find delay
    print("\nFinding delay...")
    delay = find_delay(mic, ref, sr_mic)
    print(f"  Delay: {delay} samples ({delay/sr_mic*1000:.1f} ms)")
    
    # Align and scale
    print("\nAligning and scaling reference...")
    ref_aligned, scale = align_and_scale_reference(mic, ref, delay, sr_mic)
    print(f"  Scale factor: {scale:.3f}")
    
    # Process
    print(f"\nProcessing with {args.method} method...")
    
    if args.method == 'simple':
        output = simple_subtraction(mic, ref_aligned, scale)
    elif args.method == 'spectral':
        output = frequency_domain_subtraction(mic, ref_aligned, sr_mic, scale)
    else:  # adaptive
        output = adaptive_subtraction(mic, ref_aligned, sr_mic, scale)
    
    # Normalize output to reasonable level
    voice_level = np.percentile(np.abs(output), 95)
    if voice_level > 0:
        output = output / voice_level * 0.5  # Target 50% of max
    
    print(f"\nSaving to: {args.output}")
    save_wav(args.output, sr_mic, output)
    print("Done!")

if __name__ == "__main__":
    main()







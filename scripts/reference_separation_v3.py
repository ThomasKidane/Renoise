#!/usr/bin/env python3
"""
Reference-Based Echo Cancellation v3

Properly handles the case where reference is louder than mic signal.
Uses careful scaling and frequency-domain processing.
"""

import argparse
import os
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return sr, data

def save_wav(path, sr, data):
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(path, sr, data.astype(np.float32))

def find_delay_and_scale(mic, ref, sr):
    """Find delay and scale factor between mic and reference"""
    
    # Use cross-correlation on envelopes for robustness
    # Get amplitude envelopes
    mic_env = np.abs(signal.hilbert(mic))
    ref_env = np.abs(signal.hilbert(ref))
    
    # Smooth envelopes
    window = int(sr * 0.05)  # 50ms window
    mic_env = np.convolve(mic_env, np.ones(window)/window, mode='same')
    ref_env = np.convolve(ref_env, np.ones(window)/window, mode='same')
    
    # Cross-correlation
    max_delay = int(sr * 0.5)  # 500ms max delay
    
    min_len = min(len(mic_env), len(ref_env))
    mic_env = mic_env[:min_len]
    ref_env = ref_env[:min_len]
    
    best_corr = -np.inf
    best_delay = 0
    
    for d in range(-max_delay, max_delay, 100):  # Coarse search
        if d > 0:
            m = mic_env[d:]
            r = ref_env[:len(m)]
        else:
            r = ref_env[-d:]
            m = mic_env[:len(r)]
        
        if len(m) > 0 and len(r) > 0:
            corr = np.corrcoef(m, r)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_delay = d
    
    # Fine search around best
    for d in range(best_delay - 100, best_delay + 100):
        if d > 0:
            m = mic_env[d:]
            r = ref_env[:len(m)]
        else:
            r = ref_env[-d:]
            m = mic_env[:len(r)]
        
        if len(m) > 0 and len(r) > 0:
            corr = np.corrcoef(m, r)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_delay = d
    
    print(f"  Delay: {best_delay} samples ({best_delay/sr*1000:.1f} ms)")
    print(f"  Correlation: {best_corr:.3f}")
    
    return best_delay

def align_reference(mic, ref, delay):
    """Align reference to match mic timing"""
    n = len(mic)
    ref_aligned = np.zeros(n)
    
    if delay > 0:
        # Reference needs to be shifted right (mic leads)
        copy_len = min(len(ref), n - delay)
        if copy_len > 0:
            ref_aligned[delay:delay + copy_len] = ref[:copy_len]
    elif delay < 0:
        # Reference needs to be shifted left (ref leads)
        start = -delay
        copy_len = min(len(ref) - start, n)
        if copy_len > 0 and start < len(ref):
            ref_aligned[:copy_len] = ref[start:start + copy_len]
    else:
        copy_len = min(len(ref), n)
        ref_aligned[:copy_len] = ref[:copy_len]
    
    return ref_aligned

def estimate_scale_per_band(mic, ref_aligned, sr):
    """Estimate scale factor in different frequency bands"""
    n_fft = 4096
    hop = n_fft // 4
    
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Estimate scale per frequency bin (averaged over time)
    scales = np.zeros(mic_mag.shape[0])
    
    for f in range(mic_mag.shape[0]):
        mic_band = mic_mag[f, :]
        ref_band = ref_mag[f, :]
        
        # Only use frames where reference has energy
        mask = ref_band > np.percentile(ref_band, 50)
        
        if np.sum(mask) > 10:
            # Least squares scale estimation
            ref_energy = np.sum(ref_band[mask] ** 2)
            if ref_energy > 1e-10:
                scales[f] = np.sum(mic_band[mask] * ref_band[mask]) / ref_energy
            else:
                scales[f] = 0
        else:
            scales[f] = 0
    
    # Smooth scales across frequency
    scales = signal.medfilt(scales, kernel_size=11)
    scales = np.maximum(scales, 0)
    scales = np.minimum(scales, 1.0)  # Cap at 1.0 - ref shouldn't be scaled up
    
    return scales

def spectral_subtraction_v3(mic, ref_aligned, sr, scales):
    """Spectral subtraction with per-band scaling"""
    n_fft = 4096
    hop = n_fft // 4
    
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    mic_phase = np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    output_mag = np.zeros_like(mic_mag)
    
    for t in range(mic_mag.shape[1]):
        for f in range(mic_mag.shape[0]):
            # Subtract scaled reference
            subtracted = mic_mag[f, t] - scales[f] * ref_mag[f, t]
            
            # Spectral floor - keep at least 5% of original
            output_mag[f, t] = max(subtracted, 0.05 * mic_mag[f, t])
    
    # Reconstruct
    output_stft = output_mag * np.exp(1j * mic_phase)
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def wiener_subtraction(mic, ref_aligned, sr, global_scale):
    """Wiener filter based subtraction"""
    n_fft = 4096
    hop = n_fft // 4
    
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_power = np.abs(mic_stft) ** 2
    ref_power = np.abs(ref_stft) ** 2
    mic_phase = np.angle(mic_stft)
    
    # Estimate noise (echo) power
    noise_power = (global_scale ** 2) * ref_power
    
    # Wiener gain
    signal_power = np.maximum(mic_power - noise_power, 0)
    gain = signal_power / (mic_power + 1e-10)
    
    # Smooth gain
    gain = np.sqrt(gain)  # Amplitude domain
    
    # Apply gain
    output_stft = gain * mic_stft
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', '-m', required=True)
    parser.add_argument('--ref', '-r', required=True)
    parser.add_argument('--output', '-o', required=True)
    args = parser.parse_args()
    
    print("Loading audio...")
    sr_mic, mic = load_wav(args.mic)
    sr_ref, ref = load_wav(args.ref)
    
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref ** 2))
    print(f"  Mic RMS: {mic_rms:.4f}")
    print(f"  Ref RMS: {ref_rms:.4f}")
    print(f"  Ratio: {mic_rms/ref_rms:.3f}")
    
    if sr_ref != sr_mic:
        ref = signal.resample(ref, int(len(ref) * sr_mic / sr_ref))
    
    print("\nFinding alignment...")
    delay = find_delay_and_scale(mic, ref, sr_mic)
    
    print("\nAligning reference...")
    ref_aligned = align_reference(mic, ref, delay)
    
    print("\nEstimating per-band scale factors...")
    scales = estimate_scale_per_band(mic, ref_aligned, sr_mic)
    global_scale = np.median(scales[scales > 0]) if np.any(scales > 0) else 0.2
    print(f"  Median scale: {global_scale:.3f}")
    
    print("\nApplying spectral subtraction...")
    output1 = spectral_subtraction_v3(mic, ref_aligned, sr_mic, scales)
    
    print("Applying Wiener filter...")
    output = wiener_subtraction(output1, ref_aligned, sr_mic, global_scale * 0.5)
    
    # Normalize - preserve voice level
    # Find the quieter parts (likely just voice)
    output_abs = np.abs(output)
    voice_level = np.percentile(output_abs[output_abs > 0.001], 90)
    
    if voice_level > 0:
        # Normalize so voice is at reasonable level
        output = output / voice_level * 0.3
    
    # Final clip
    output = np.clip(output, -1.0, 1.0)
    
    print(f"\nSaving to: {args.output}")
    save_wav(args.output, sr_mic, output)
    print("Done!")

if __name__ == "__main__":
    main()



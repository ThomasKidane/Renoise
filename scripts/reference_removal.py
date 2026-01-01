#!/usr/bin/env python3
"""
Reference Signal Removal - Keep ONLY the speaker's voice

The goal: Remove EVERYTHING from the reference (music, lyrics, sounds from computer)
and keep ONLY the voice speaking into the microphone.

This is different from:
- Speech separation (keeps all speech including lyrics)
- Speech enhancement (removes noise but keeps speech)

We need: Echo cancellation that removes the EXACT reference signal.
"""

import numpy as np
import os
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

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
    sf.write(path, data, sr)

def find_best_alignment(mic, ref, sr, search_range_ms=500):
    """Find the best alignment between mic and reference using cross-correlation"""
    search_samples = int(search_range_ms * sr / 1000)
    
    # Use shorter chunk for speed
    chunk_len = min(sr * 3, len(mic), len(ref))
    mic_chunk = mic[:chunk_len]
    ref_chunk = ref[:chunk_len]
    
    best_corr = -np.inf
    best_delay = 0
    best_scale = 1.0
    
    # Coarse search
    for delay in range(-search_samples, search_samples, 50):
        if delay > 0:
            m = mic_chunk[delay:]
            r = ref_chunk[:len(m)]
        else:
            r = ref_chunk[-delay:]
            m = mic_chunk[:len(r)]
        
        if len(m) < 1000:
            continue
            
        # Find optimal scale
        scale = np.dot(m, r) / (np.dot(r, r) + 1e-10)
        scale = np.clip(scale, 0, 2)
        
        # Compute correlation after scaling
        residual = m - scale * r
        corr = 1 - np.var(residual) / (np.var(m) + 1e-10)
        
        if corr > best_corr:
            best_corr = corr
            best_delay = delay
            best_scale = scale
    
    # Fine search around best
    for delay in range(best_delay - 50, best_delay + 50):
        if delay > 0:
            m = mic_chunk[delay:]
            r = ref_chunk[:len(m)]
        else:
            r = ref_chunk[-delay:]
            m = mic_chunk[:len(r)]
        
        if len(m) < 1000:
            continue
            
        scale = np.dot(m, r) / (np.dot(r, r) + 1e-10)
        scale = np.clip(scale, 0, 2)
        
        residual = m - scale * r
        corr = 1 - np.var(residual) / (np.var(m) + 1e-10)
        
        if corr > best_corr:
            best_corr = corr
            best_delay = delay
            best_scale = scale
    
    return best_delay, best_scale, best_corr

def align_reference(mic, ref, delay):
    """Shift reference to align with mic"""
    n = len(mic)
    aligned = np.zeros(n)
    
    if delay >= 0:
        copy_len = min(len(ref), n - delay)
        if copy_len > 0:
            aligned[delay:delay + copy_len] = ref[:copy_len]
    else:
        start = -delay
        copy_len = min(len(ref) - start, n)
        if copy_len > 0 and start < len(ref):
            aligned[:copy_len] = ref[start:start + copy_len]
    
    return aligned

def frequency_domain_subtraction(mic, ref_aligned, sr, scale, over_sub=1.0):
    """
    Subtract reference in frequency domain with proper phase alignment
    """
    n_fft = 2048
    hop = n_fft // 4
    
    # STFT
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    # Complex subtraction (preserves phase relationships)
    output_stft = mic_stft - over_sub * scale * ref_stft
    
    # Reconstruct
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def adaptive_frequency_subtraction(mic, ref_aligned, sr):
    """
    Adaptive subtraction that estimates scale per frequency band
    """
    n_fft = 4096
    hop = n_fft // 4
    
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    output_stft = np.zeros_like(mic_stft)
    
    # Estimate scale per frequency band
    for freq_bin in range(mic_stft.shape[0]):
        mic_band = mic_stft[freq_bin, :]
        ref_band = ref_stft[freq_bin, :]
        
        # Find scale that minimizes residual energy
        ref_power = np.sum(np.abs(ref_band) ** 2)
        if ref_power > 1e-10:
            # Complex scale estimation
            scale = np.sum(mic_band * np.conj(ref_band)) / ref_power
            # Limit scale magnitude
            scale = np.clip(np.abs(scale), 0, 2) * np.exp(1j * np.angle(scale))
        else:
            scale = 0
        
        output_stft[freq_bin, :] = mic_band - scale * ref_band
    
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    return output[:len(mic)]

def nlms_echo_cancellation(mic, ref, filter_len=8192, mu=0.1):
    """
    NLMS adaptive filter - learns the acoustic path from speaker to mic
    and subtracts the estimated echo
    """
    n = len(mic)
    
    # Ensure same length
    if len(ref) < n:
        ref = np.pad(ref, (0, n - len(ref)))
    else:
        ref = ref[:n]
    
    # Initialize filter
    w = np.zeros(filter_len)
    output = np.zeros(n)
    
    # Process sample by sample
    for i in range(filter_len, n):
        # Get reference buffer (reversed for convolution)
        x = ref[i-filter_len:i][::-1]
        
        # Estimate echo
        echo_estimate = np.dot(w, x)
        
        # Output = mic - echo estimate
        e = mic[i] - echo_estimate
        output[i] = e
        
        # Update filter (NLMS)
        norm = np.dot(x, x) + 1e-10
        w = w + (mu / norm) * e * x
    
    return output

def multi_delay_cancellation(mic, ref, sr):
    """
    Try multiple delays and combine results
    Sometimes the acoustic path has multiple reflections
    """
    # Find primary delay
    delay1, scale1, _ = find_best_alignment(mic, ref, sr)
    ref1 = align_reference(mic, ref, delay1)
    
    # Subtract primary
    residual1 = mic - scale1 * ref1
    
    # Find secondary delay in residual
    delay2, scale2, corr2 = find_best_alignment(residual1, ref, sr)
    
    if corr2 > 0.1:  # If there's significant correlation
        ref2 = align_reference(mic, ref, delay2)
        residual2 = residual1 - scale2 * 0.5 * ref2  # Use smaller scale for secondary
        return residual2
    
    return residual1

def main():
    mic_path = '../recordings/raw_input_2025-12-31T15-29-41-535Z.wav'
    ref_path = '../recordings/reference_2025-12-31T15-29-41-535Z.wav'
    out_dir = '../recordings/samples'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading audio...")
    sr, mic = load_wav(mic_path)
    _, ref = load_wav(ref_path)
    
    print(f"  Mic: {len(mic)/sr:.1f}s @ {sr}Hz, RMS={np.sqrt(np.mean(mic**2)):.4f}")
    print(f"  Ref: {len(ref)/sr:.1f}s @ {sr}Hz, RMS={np.sqrt(np.mean(ref**2)):.4f}")
    
    # Find alignment
    print("\nFinding optimal alignment...")
    delay, scale, corr = find_best_alignment(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay/sr*1000:.1f} ms)")
    print(f"  Scale: {scale:.4f}")
    print(f"  Correlation: {corr:.4f}")
    
    ref_aligned = align_reference(mic, ref, delay)
    
    methods = [
        ("ref_time_subtract", lambda: mic - scale * ref_aligned),
        ("ref_freq_subtract", lambda: frequency_domain_subtraction(mic, ref_aligned, sr, scale, 1.0)),
        ("ref_freq_over_subtract", lambda: frequency_domain_subtraction(mic, ref_aligned, sr, scale, 1.2)),
        ("ref_adaptive_freq", lambda: adaptive_frequency_subtraction(mic, ref_aligned, sr)),
        ("ref_nlms_8k", lambda: nlms_echo_cancellation(mic, ref_aligned, 8192, 0.1)),
        ("ref_nlms_16k", lambda: nlms_echo_cancellation(mic, ref_aligned, 16384, 0.05)),
        ("ref_multi_delay", lambda: multi_delay_cancellation(mic, ref, sr)),
    ]
    
    for name, method in methods:
        print(f"\nProcessing: {name}...")
        try:
            output = method()
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * 0.7
            
            out_path = os.path.join(out_dir, f"{name}.wav")
            save_wav(out_path, sr, output)
            print(f"  ✓ Saved: {out_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✅ Done! Check recordings/samples/ for outputs starting with 'ref_'")

if __name__ == "__main__":
    main()



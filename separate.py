#!/usr/bin/env python3
"""
Perfect Audio Separation using Phase Cancellation

When you have the EXACT reference signal that was mixed into the recording,
phase cancellation is the mathematically perfect solution.

Usage:
    python3 separate.py <mixed_input.wav> <reference.wav> <output.wav> [delay_samples]

The script will:
1. Align the reference to the mixed signal (auto-detect delay if not specified)
2. Subtract the reference using phase cancellation
3. Output the separated voice
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

def load_wav(path):
    """Load WAV file and convert to float32 mono"""
    rate, data = wavfile.read(path)
    
    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    return rate, data

def save_wav(path, rate, data):
    """Save as 16-bit WAV"""
    # Normalize to prevent clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.95
    
    # Convert to int16
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(path, rate, data_int16)

def find_delay(mixed, reference, max_delay=48000):
    """Find the delay between reference and mixed signal using cross-correlation"""
    # Use a chunk for faster processing
    chunk_size = min(len(mixed), len(reference), 48000 * 5)  # 5 seconds max
    
    mixed_chunk = mixed[:chunk_size]
    ref_chunk = reference[:chunk_size]
    
    # Cross-correlation
    correlation = correlate(mixed_chunk, ref_chunk, mode='full')
    
    # Find the peak
    center = len(ref_chunk) - 1
    search_range = min(max_delay, center)
    
    start_idx = center - search_range
    end_idx = center + search_range
    
    peak_idx = start_idx + np.argmax(correlation[start_idx:end_idx])
    delay = peak_idx - center
    
    # Confidence score
    peak_val = correlation[peak_idx]
    mean_val = np.mean(np.abs(correlation[start_idx:end_idx]))
    confidence = peak_val / mean_val if mean_val > 0 else 0
    
    return delay, confidence

def find_gain(mixed, reference, delay):
    """Find the optimal gain to match reference level in mixed signal"""
    # Align signals
    if delay >= 0:
        ref_aligned = reference[delay:]
        mix_aligned = mixed[:len(ref_aligned)]
    else:
        mix_aligned = mixed[-delay:]
        ref_aligned = reference[:len(mix_aligned)]
    
    min_len = min(len(mix_aligned), len(ref_aligned))
    mix_aligned = mix_aligned[:min_len]
    ref_aligned = ref_aligned[:min_len]
    
    # Optimal gain using least squares
    ref_power = np.sum(ref_aligned ** 2)
    if ref_power > 0:
        gain = np.sum(mix_aligned * ref_aligned) / ref_power
    else:
        gain = 1.0
    
    return gain

def phase_cancel(mixed, reference, delay, gain):
    """Subtract reference from mixed signal"""
    output = np.zeros_like(mixed)
    
    if delay >= 0:
        # Reference is delayed relative to mixed
        ref_len = min(len(reference), len(mixed) - delay)
        output[:delay] = mixed[:delay]  # Keep beginning
        output[delay:delay+ref_len] = mixed[delay:delay+ref_len] - gain * reference[:ref_len]
        if delay + ref_len < len(mixed):
            output[delay+ref_len:] = mixed[delay+ref_len:]  # Keep end
    else:
        # Mixed is delayed relative to reference
        abs_delay = -delay
        mix_len = min(len(mixed), len(reference) - abs_delay)
        output[:mix_len] = mixed[:mix_len] - gain * reference[abs_delay:abs_delay+mix_len]
        if mix_len < len(mixed):
            output[mix_len:] = mixed[mix_len:]
    
    return output

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 separate.py <mixed_input.wav> <reference.wav> <output.wav> [delay_samples]")
        print("\nThis script removes the reference audio from the mixed input using phase cancellation.")
        print("Works best when the reference is the EXACT audio that was mixed in.")
        sys.exit(1)
    
    mixed_path = sys.argv[1]
    ref_path = sys.argv[2]
    output_path = sys.argv[3]
    manual_delay = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    print(f"Loading mixed audio: {mixed_path}")
    mix_rate, mixed = load_wav(mixed_path)
    
    print(f"Loading reference audio: {ref_path}")
    ref_rate, reference = load_wav(ref_path)
    
    if mix_rate != ref_rate:
        print(f"Warning: Sample rates differ ({mix_rate} vs {ref_rate})")
        print("Resampling reference to match mixed...")
        from scipy.signal import resample
        reference = resample(reference, int(len(reference) * mix_rate / ref_rate))
    
    print(f"Mixed: {len(mixed)} samples ({len(mixed)/mix_rate:.2f}s)")
    print(f"Reference: {len(reference)} samples ({len(reference)/mix_rate:.2f}s)")
    
    # Find delay
    if manual_delay is not None:
        delay = manual_delay
        print(f"Using manual delay: {delay} samples ({delay/mix_rate*1000:.1f}ms)")
    else:
        print("Finding optimal delay...")
        delay, confidence = find_delay(mixed, reference)
        print(f"Detected delay: {delay} samples ({delay/mix_rate*1000:.1f}ms), confidence: {confidence:.2f}")
    
    # Find optimal gain
    print("Finding optimal gain...")
    gain = find_gain(mixed, reference, delay)
    print(f"Optimal gain: {gain:.4f}")
    
    # Phase cancellation
    print("Applying phase cancellation...")
    output = phase_cancel(mixed, reference, delay, gain)
    
    # Calculate reduction
    mixed_rms = np.sqrt(np.mean(mixed ** 2))
    output_rms = np.sqrt(np.mean(output ** 2))
    reduction_db = 20 * np.log10(output_rms / mixed_rms) if mixed_rms > 0 else 0
    
    print(f"RMS reduction: {reduction_db:.1f} dB")
    
    # Save output
    print(f"Saving output: {output_path}")
    save_wav(output_path, mix_rate, output)
    
    print("Done!")

if __name__ == "__main__":
    main()


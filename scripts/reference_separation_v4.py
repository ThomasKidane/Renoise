#!/usr/bin/env python3
"""
Reference-Based Echo Cancellation v4

Uses masking instead of subtraction to avoid distortion artifacts.
The idea: use the reference to identify WHERE the music is, then 
attenuate those regions while preserving voice.
"""

import argparse
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

def find_delay(mic, ref, sr):
    """Find delay using envelope correlation"""
    # Get envelopes
    mic_env = np.abs(signal.hilbert(mic))
    ref_env = np.abs(signal.hilbert(ref))
    
    # Smooth
    win = int(sr * 0.02)
    mic_env = np.convolve(mic_env, np.ones(win)/win, mode='same')
    ref_env = np.convolve(ref_env, np.ones(win)/win, mode='same')
    
    # Cross-correlation
    corr = signal.correlate(mic_env, ref_env, mode='full')
    lags = signal.correlation_lags(len(mic_env), len(ref_env), mode='full')
    
    # Find peak within reasonable range
    max_delay = int(sr * 0.3)  # 300ms
    valid = np.abs(lags) < max_delay
    corr_valid = corr.copy()
    corr_valid[~valid] = 0
    
    best_idx = np.argmax(corr_valid)
    delay = lags[best_idx]
    
    return delay

def align_reference(mic, ref, delay):
    """Align reference to mic"""
    n = len(mic)
    ref_aligned = np.zeros(n)
    
    if delay >= 0:
        copy_len = min(len(ref), n - delay)
        if copy_len > 0:
            ref_aligned[delay:delay + copy_len] = ref[:copy_len]
    else:
        start = -delay
        copy_len = min(len(ref) - start, n)
        if copy_len > 0 and start < len(ref):
            ref_aligned[:copy_len] = ref[start:start + copy_len]
    
    return ref_aligned

def compute_voice_mask(mic, ref_aligned, sr):
    """
    Compute a soft mask that identifies voice vs music.
    
    Strategy: 
    - Where reference has high energy, music is present
    - Where mic has energy but reference doesn't, voice is present
    - Use ratio to create soft mask
    """
    n_fft = 2048
    hop = n_fft // 4
    
    # STFT
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    mic_phase = np.angle(mic_stft)
    
    # Estimate the scale factor (how loud is music in mic vs reference)
    # Use median ratio in high-energy frames
    ref_energy = np.sum(ref_mag ** 2, axis=0)
    high_energy_frames = ref_energy > np.percentile(ref_energy, 70)
    
    if np.sum(high_energy_frames) > 5:
        mic_high = mic_mag[:, high_energy_frames]
        ref_high = ref_mag[:, high_energy_frames]
        
        # Per-bin scale estimation
        scales = np.zeros(mic_mag.shape[0])
        for f in range(len(scales)):
            ref_f = ref_high[f, :]
            mic_f = mic_high[f, :]
            if np.sum(ref_f ** 2) > 1e-10:
                scales[f] = np.sum(mic_f * ref_f) / np.sum(ref_f ** 2)
        
        scales = np.clip(scales, 0, 0.5)  # Music in mic is at most 50% of reference
        scales = signal.medfilt(scales, 5)  # Smooth
    else:
        scales = np.ones(mic_mag.shape[0]) * 0.2
    
    # Create mask: high where voice likely, low where music likely
    # mask = 1 - (scaled_ref / mic)  but bounded
    
    mask = np.ones_like(mic_mag)
    
    for t in range(mic_mag.shape[1]):
        for f in range(mic_mag.shape[0]):
            mic_val = mic_mag[f, t]
            ref_val = scales[f] * ref_mag[f, t]
            
            if mic_val > 1e-10:
                # Ratio of estimated music to total
                music_ratio = ref_val / mic_val
                music_ratio = min(music_ratio, 1.0)
                
                # Mask: suppress where music dominates
                # Use soft transition
                mask[f, t] = max(0, 1 - music_ratio ** 0.5)
            else:
                mask[f, t] = 1.0
    
    return mic_stft, mask, mic_phase

def apply_mask_with_smoothing(mic_stft, mask, mic_phase, sr):
    """Apply mask with temporal and spectral smoothing to reduce artifacts"""
    n_fft = mic_stft.shape[0] * 2 - 2  # Recover n_fft from STFT shape
    hop = n_fft // 4
    
    # Smooth mask temporally
    for f in range(mask.shape[0]):
        mask[f, :] = signal.medfilt(mask[f, :], kernel_size=5)
    
    # Smooth mask spectrally
    for t in range(mask.shape[1]):
        mask[:, t] = signal.medfilt(mask[:, t], kernel_size=3)
    
    # Apply minimum mask value to avoid complete silence
    mask = np.maximum(mask, 0.1)
    
    # Apply mask
    output_stft = mask * mic_stft
    
    # Reconstruct
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output

def time_domain_cleanup(output, ref_aligned, sr, scale=0.15):
    """
    Light time-domain subtraction for any remaining coherent music
    Uses very conservative scaling to avoid artifacts
    """
    # Only subtract a small amount
    cleaned = output - scale * ref_aligned[:len(output)]
    return cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', '-m', required=True)
    parser.add_argument('--ref', '-r', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--aggressive', type=float, default=0.5,
                        help='Aggressiveness 0-1 (higher = more music removal but more artifacts)')
    args = parser.parse_args()
    
    print("Loading audio...")
    sr_mic, mic = load_wav(args.mic)
    sr_ref, ref = load_wav(args.ref)
    
    print(f"  Mic: {len(mic)/sr_mic:.1f}s, RMS={np.sqrt(np.mean(mic**2)):.4f}")
    print(f"  Ref: {len(ref)/sr_ref:.1f}s, RMS={np.sqrt(np.mean(ref**2)):.4f}")
    
    if sr_ref != sr_mic:
        ref = signal.resample(ref, int(len(ref) * sr_mic / sr_ref))
    
    print("\nFinding alignment...")
    delay = find_delay(mic, ref, sr_mic)
    print(f"  Delay: {delay} samples ({delay/sr_mic*1000:.1f} ms)")
    
    print("\nAligning reference...")
    ref_aligned = align_reference(mic, ref, delay)
    
    print("\nComputing voice/music mask...")
    mic_stft, mask, mic_phase = compute_voice_mask(mic, ref_aligned, sr_mic)
    
    # Adjust mask based on aggressiveness
    mask = mask ** args.aggressive  # Higher power = more aggressive masking
    
    print("\nApplying mask...")
    output = apply_mask_with_smoothing(mic_stft, mask, mic_phase, sr_mic)
    
    # Trim to original length
    output = output[:len(mic)]
    
    # Optional: light time-domain cleanup
    # print("\nTime-domain cleanup...")
    # output = time_domain_cleanup(output, ref_aligned, sr_mic, scale=0.1)
    
    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.7
    
    print(f"\nSaving to: {args.output}")
    save_wav(args.output, sr_mic, output)
    print("Done!")

if __name__ == "__main__":
    main()



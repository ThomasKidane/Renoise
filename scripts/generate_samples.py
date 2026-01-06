#!/usr/bin/env python3
"""
Generate multiple separation samples using different methods
for comparison.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import os

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
    """Find delay using cross-correlation"""
    chunk = min(sr * 5, len(mic), len(ref))
    corr = signal.correlate(mic[:chunk], ref[:chunk], mode='full')
    lags = signal.correlation_lags(chunk, chunk, mode='full')
    
    max_delay = int(sr * 0.3)
    valid = np.abs(lags) < max_delay
    corr[~valid] = 0
    
    return lags[np.argmax(np.abs(corr))]

def align_ref(mic, ref, delay):
    n = len(mic)
    aligned = np.zeros(n)
    if delay >= 0:
        copy_len = min(len(ref), n - delay)
        if copy_len > 0:
            aligned[delay:delay+copy_len] = ref[:copy_len]
    else:
        start = -delay
        copy_len = min(len(ref) - start, n)
        if copy_len > 0:
            aligned[:copy_len] = ref[start:start+copy_len]
    return aligned

# ============ METHOD 1: Simple scaled subtraction ============
def method_simple_subtract(mic, ref_aligned, scale):
    """Direct time-domain subtraction"""
    output = mic - scale * ref_aligned
    return output

# ============ METHOD 2: High-pass filter (remove bass where music lives) ============
def method_highpass(mic, sr, cutoff=300):
    """High-pass filter to remove low frequencies where music dominates"""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, mic)

# ============ METHOD 3: Voice frequency emphasis ============
def method_voice_emphasis(mic, sr):
    """Boost voice frequencies (300-3000 Hz), cut others"""
    # Bandpass for voice
    sos = signal.butter(4, [200, 4000], btype='band', fs=sr, output='sos')
    voice_band = signal.sosfilt(sos, mic)
    return voice_band

# ============ METHOD 4: Spectral gate based on reference ============
def method_spectral_gate(mic, ref_aligned, sr, threshold=0.3):
    """Gate frequencies where reference has energy"""
    n_fft = 2048
    hop = n_fft // 4
    
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Normalize reference
    ref_norm = ref_mag / (np.max(ref_mag) + 1e-10)
    
    # Create gate: close where reference is loud
    gate = 1.0 - np.clip(ref_norm / threshold, 0, 1)
    
    # Smooth gate
    for f in range(gate.shape[0]):
        gate[f, :] = signal.medfilt(gate[f, :], 5)
    
    output_stft = gate * mic_stft
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# ============ METHOD 5: Wiener filter with reference as noise estimate ============
def method_wiener(mic, ref_aligned, sr, noise_scale=0.2):
    """Wiener filter treating scaled reference as noise"""
    n_fft = 2048
    hop = n_fft // 4
    
    _, _, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_power = np.abs(mic_stft) ** 2
    noise_power = (noise_scale ** 2) * np.abs(ref_stft) ** 2
    
    # Wiener gain
    gain = np.maximum(mic_power - noise_power, 0) / (mic_power + 1e-10)
    gain = np.sqrt(gain)
    
    output_stft = gain * mic_stft
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# ============ METHOD 6: NLMS Adaptive Filter ============
def method_nlms(mic, ref_aligned, filter_len=4096, mu=0.1):
    """NLMS adaptive filter"""
    n = len(mic)
    w = np.zeros(filter_len)
    output = np.zeros(n)
    
    ref_padded = np.pad(ref_aligned, (filter_len, 0))
    
    for i in range(n):
        x = ref_padded[i:i+filter_len][::-1]
        y_hat = np.dot(w, x)
        e = mic[i] - y_hat
        output[i] = e
        
        norm = np.dot(x, x) + 1e-10
        w = w + (mu / norm) * e * x
    
    return output

# ============ METHOD 7: Combined approach ============
def method_combined(mic, ref_aligned, sr):
    """Combine NLMS + Wiener + bandpass"""
    # Step 1: NLMS
    nlms_out = method_nlms(mic, ref_aligned, filter_len=2048, mu=0.05)
    
    # Step 2: Wiener cleanup
    wiener_out = method_wiener(nlms_out, ref_aligned, sr, noise_scale=0.1)
    
    # Step 3: Voice bandpass
    sos = signal.butter(2, [200, 4000], btype='band', fs=sr, output='sos')
    final = signal.sosfilt(sos, wiener_out)
    
    return final

def normalize(audio, target_rms=0.1):
    """Normalize audio to target RMS"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-10:
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)

def main():
    mic_path = '../recordings/raw_input_2025-12-31T15-29-41-535Z.wav'
    ref_path = '../recordings/reference_2025-12-31T15-29-41-535Z.wav'
    out_dir = '../recordings/samples'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading audio...")
    sr, mic = load_wav(mic_path)
    _, ref = load_wav(ref_path)
    
    print("Finding alignment...")
    delay = find_delay(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay/sr*1000:.1f} ms)")
    
    ref_aligned = align_ref(mic, ref, delay)
    
    # Estimate scale
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref_aligned ** 2))
    scale = mic_rms / ref_rms
    print(f"  Scale: {scale:.3f}")
    
    methods = [
        ("1_simple_subtract", lambda: method_simple_subtract(mic, ref_aligned, scale)),
        ("2_highpass_300hz", lambda: method_highpass(mic, sr, 300)),
        ("3_voice_bandpass", lambda: method_voice_emphasis(mic, sr)),
        ("4_spectral_gate", lambda: method_spectral_gate(mic, ref_aligned, sr, 0.2)),
        ("5_wiener_filter", lambda: method_wiener(mic, ref_aligned, sr, scale)),
        ("6_nlms_adaptive", lambda: method_nlms(mic, ref_aligned, 4096, 0.1)),
        ("7_combined", lambda: method_combined(mic, ref_aligned, sr)),
    ]
    
    for name, method in methods:
        print(f"\nProcessing: {name}...")
        try:
            output = method()
            output = normalize(output, 0.15)
            out_path = os.path.join(out_dir, f"{name}.wav")
            save_wav(out_path, sr, output)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✅ All samples generated in recordings/samples/")
    print("Listen to each and let me know which sounds best!")

if __name__ == "__main__":
    main()







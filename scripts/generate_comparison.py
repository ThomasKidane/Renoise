#!/usr/bin/env python3
"""
Generate multiple speaker separation samples using different methods
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
import os
import torch


def load_audio(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def align_signals(mic, ref, sr):
    chunk = min(sr * 2, len(mic))
    corr = signal.correlate(mic[:chunk], ref[:chunk], mode='same')
    delay = np.argmax(corr) - chunk // 2
    
    if delay > 0:
        ref = np.concatenate([np.zeros(delay, dtype=np.float32), ref])[:len(mic)]
    elif delay < 0:
        ref = ref[-delay:]
    if len(ref) < len(mic):
        ref = np.concatenate([ref, np.zeros(len(mic) - len(ref), dtype=np.float32)])
    return ref[:len(mic)].astype(np.float32), delay


def run_demucs(audio, orig_sr):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = get_model('htdemucs')
    model.to(device)
    model.eval()
    
    audio_44k = signal.resample_poly(audio, 44100, orig_sr).astype(np.float32)
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1).to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=False)
    
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    vocals = signal.resample_poly(vocals, orig_sr, 44100).astype(np.float32)
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    return vocals


def highpass(audio, sr, cutoff=100):
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def normalize(audio):
    peak = np.abs(audio).max()
    if peak > 0:
        return (audio / peak * 0.9).astype(np.float32)
    return audio


# ============ METHOD 1: Simple Spectral Subtraction ============
def method_spectral_subtract(mic_vocals, ref_vocals, sr):
    """Simple spectral subtraction - subtract reference magnitude from mic."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Simple subtraction with floor
    alpha = 1.5  # Over-subtraction factor
    clean_mag = np.maximum(mic_mag - alpha * ref_mag, 0.05 * mic_mag)
    
    clean_stft = clean_mag * np.exp(1j * mic_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 2: Wiener Filter ============
def method_wiener(mic_vocals, ref_vocals, sr):
    """Wiener filter - optimal linear filter."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    eps = 1e-8
    # Wiener gain: signal / (signal + noise)
    # Here "noise" is the singer
    your_mag_est = np.maximum(mic_mag - ref_mag, 0)
    gain = your_mag_est ** 2 / (your_mag_est ** 2 + ref_mag ** 2 + eps)
    gain = np.sqrt(gain)
    
    clean_stft = mic_mag * gain * np.exp(1j * mic_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 3: Ideal Ratio Mask ============
def method_irm(mic_vocals, ref_vocals, sr):
    """Ideal Ratio Mask - soft mask based on magnitude ratio."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    eps = 1e-8
    # IRM: target / (target + interference)
    your_mag_est = np.maximum(mic_mag - ref_mag, 0)
    mask = your_mag_est / (mic_mag + eps)
    mask = np.clip(mask, 0, 1)
    
    clean_stft = mic_mag * mask * np.exp(1j * mic_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 4: Binary Mask ============
def method_binary_mask(mic_vocals, ref_vocals, sr):
    """Binary mask - hard decision based on which is louder."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Binary: 1 where your voice dominates, 0 where singer dominates
    your_mag_est = np.maximum(mic_mag - ref_mag, 0)
    mask = (your_mag_est > ref_mag).astype(np.float32)
    
    # Smooth slightly
    mask = median_filter(mask, size=(3, 3))
    
    clean_stft = mic_mag * mask * np.exp(1j * mic_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 5: Phase-Sensitive Mask ============
def method_psm(mic_vocals, ref_vocals, sr):
    """Phase-Sensitive Mask - uses phase information."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
    ref_mag, ref_phase = np.abs(ref_stft), np.angle(ref_stft)
    
    eps = 1e-8
    
    # Estimate your voice using phase difference
    phase_diff = np.cos(mic_phase - ref_phase)
    
    # Where phases align, it's likely the singer
    # Where phases differ, it's likely you
    your_mag_est = mic_mag * (1 - 0.5 * (phase_diff + 1))  # 0 when aligned, 1 when opposite
    
    mask = your_mag_est / (mic_mag + eps)
    mask = np.clip(mask, 0.1, 1)
    
    clean_stft = mic_mag * mask * np.exp(1j * mic_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 6: Adaptive Transfer Function ============
def method_adaptive_tf(mic_vocals, ref_vocals, sr):
    """Adaptive transfer function estimation and subtraction."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    n_freq, n_frames = mic_stft.shape
    eps = 1e-8
    
    # Time-varying transfer function
    H = np.zeros((n_freq, n_frames), dtype=np.complex64)
    alpha = 0.95
    num = np.zeros(n_freq, dtype=np.complex64)
    den = np.zeros(n_freq, dtype=np.float32)
    
    for t in range(n_frames):
        num = alpha * num + (1 - alpha) * mic_stft[:, t] * np.conj(ref_stft[:, t])
        den = alpha * den + (1 - alpha) * np.abs(ref_stft[:, t]) ** 2
        H[:, t] = num / (den + eps)
    
    # Subtract
    singer_est = H * ref_stft
    your_stft = mic_stft - singer_est
    
    _, clean = signal.istft(your_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 7: Complex Ratio Mask ============
def method_crm(mic_vocals, ref_vocals, sr):
    """Complex Ratio Mask - operates on complex STFT."""
    n_fft, hop = 1024, 256
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_f = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
    
    eps = 1e-8
    
    # Estimate transfer function
    H = np.sum(mic_stft * np.conj(ref_stft), axis=1) / (np.sum(np.abs(ref_stft)**2, axis=1) + eps)
    H = H[:, np.newaxis]
    
    # Estimate singer
    singer_est = H * ref_stft
    
    # Complex subtraction
    your_stft = mic_stft - singer_est
    
    # Apply soft mask for cleanup
    your_mag = np.abs(your_stft)
    singer_mag = np.abs(singer_est)
    mask = your_mag / (your_mag + singer_mag + eps)
    
    clean_stft = your_stft * mask
    
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return clean[:len(mic_vocals)].astype(np.float32)


# ============ METHOD 8: Multi-Resolution ============
def method_multires(mic_vocals, ref_vocals, sr):
    """Multi-resolution processing - combine multiple FFT sizes."""
    results = []
    
    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        
        _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        min_f = min(mic_stft.shape[1], ref_stft.shape[1])
        mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
        
        mic_mag, mic_phase = np.abs(mic_stft), np.angle(mic_stft)
        ref_mag = np.abs(ref_stft)
        
        eps = 1e-8
        your_mag_est = np.maximum(mic_mag - ref_mag, 0)
        mask = your_mag_est / (mic_mag + eps)
        
        clean_stft = mic_mag * mask * np.exp(1j * mic_phase)
        _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        clean = clean[:len(mic_vocals)]
        if len(clean) < len(mic_vocals):
            clean = np.concatenate([clean, np.zeros(len(mic_vocals) - len(clean))])
        results.append(clean)
    
    return np.mean(results, axis=0).astype(np.float32)


def main():
    import sys
    
    mic_path = "recordings/raw_input_2025-12-31T15-29-41-535Z.wav"
    ref_path = "recordings/reference_2025-12-31T15-29-41-535Z.wav"
    output_dir = "recordings/samples/comparison"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Speaker Separation Comparison Samples")
    print("=" * 60)
    
    # Load
    print("\n[1/3] Loading audio...")
    mic, sr = load_audio(mic_path, 16000)
    ref, _ = load_audio(ref_path, 16000)
    
    min_len = min(len(mic), len(ref))
    mic, ref = mic[:min_len], ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s at {sr}Hz")
    
    # Align
    print("\n[2/3] Aligning and extracting vocals...")
    ref_aligned, delay = align_signals(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay*1000/sr:.1f}ms)")
    
    # Extract vocals with Demucs
    print("  Extracting vocals from mic...")
    mic_vocals = run_demucs(mic, sr)
    print("  Extracting vocals from reference...")
    ref_vocals = run_demucs(ref_aligned, sr)
    
    # Generate samples
    print("\n[3/3] Generating comparison samples...")
    
    methods = [
        ("1_spectral_subtract", method_spectral_subtract, "Simple spectral subtraction"),
        ("2_wiener", method_wiener, "Wiener filter"),
        ("3_irm", method_irm, "Ideal Ratio Mask"),
        ("4_binary_mask", method_binary_mask, "Binary mask"),
        ("5_psm", method_psm, "Phase-Sensitive Mask"),
        ("6_adaptive_tf", method_adaptive_tf, "Adaptive transfer function"),
        ("7_crm", method_crm, "Complex Ratio Mask"),
        ("8_multires", method_multires, "Multi-resolution"),
    ]
    
    for name, method, desc in methods:
        print(f"  {desc}...")
        try:
            result = method(mic_vocals, ref_vocals, sr)
            result = highpass(result, sr, 100)
            result = normalize(result)
            
            output_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(output_path, result, sr)
            print(f"    ✓ {output_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # Also save the Demucs vocals for reference
    sf.write(os.path.join(output_dir, "0_demucs_vocals.wav"), normalize(highpass(mic_vocals, sr, 100)), sr)
    print(f"    ✓ {output_dir}/0_demucs_vocals.wav (Demucs vocals only)")
    
    print("\n" + "=" * 60)
    print(f"Done! Check {output_dir}/ for comparison samples")
    print("=" * 60)


if __name__ == '__main__':
    main()



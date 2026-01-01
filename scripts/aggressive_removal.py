#!/usr/bin/env python3
"""
Aggressive Reference Removal

Since the correlation is low, we need to be more aggressive about
finding and removing the reference signal.

This version:
1. Tries multiple alignment strategies
2. Uses room impulse response estimation
3. Applies aggressive spectral subtraction
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

def estimate_room_impulse_response(mic, ref, sr, ir_length=4096):
    """
    Estimate the room impulse response using Wiener-Hopf equation
    This models how the reference sounds after going through speakers and room
    """
    # Use shorter signals for computation
    n = min(len(mic), len(ref), sr * 10)  # 10 seconds max
    mic_chunk = mic[:n]
    ref_chunk = ref[:n]
    
    # Compute cross-correlation and auto-correlation
    # R_rr * h = R_rm  (Wiener-Hopf equation)
    
    # Auto-correlation of reference
    R_rr = np.correlate(ref_chunk, ref_chunk, mode='full')
    center = len(R_rr) // 2
    R_rr = R_rr[center:center + ir_length]
    
    # Cross-correlation between mic and reference
    R_rm = np.correlate(mic_chunk, ref_chunk, mode='full')
    center = len(R_rm) // 2
    R_rm = R_rm[center:center + ir_length]
    
    # Build Toeplitz matrix for R_rr
    from scipy.linalg import toeplitz, solve_toeplitz
    
    # Solve for impulse response using Levinson-Durbin
    try:
        # Add regularization
        R_rr[0] += 1e-6 * R_rr[0]
        h = solve_toeplitz(R_rr, R_rm)
    except:
        h = np.zeros(ir_length)
        h[0] = R_rm[0] / (R_rr[0] + 1e-10)
    
    return h

def convolve_and_subtract(mic, ref, impulse_response):
    """Apply estimated room impulse response to reference and subtract"""
    # Convolve reference with impulse response
    echo_estimate = signal.convolve(ref, impulse_response, mode='full')[:len(mic)]
    
    # Ensure same length
    if len(echo_estimate) < len(mic):
        echo_estimate = np.pad(echo_estimate, (0, len(mic) - len(echo_estimate)))
    
    # Subtract
    output = mic - echo_estimate
    return output

def spectral_subtraction_aggressive(mic, ref, sr, alpha=2.0, beta=0.01):
    """
    Very aggressive spectral subtraction
    alpha > 1 means we subtract MORE than the estimated noise
    """
    n_fft = 2048
    hop = n_fft // 4
    
    # Ensure same length
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    mic_phase = np.angle(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Estimate per-frame scale
    output_mag = np.zeros_like(mic_mag)
    
    for t_idx in range(mic_mag.shape[1]):
        mic_frame = mic_mag[:, t_idx]
        ref_frame = ref_mag[:, t_idx]
        
        # Estimate scale for this frame
        ref_energy = np.sum(ref_frame ** 2)
        if ref_energy > 1e-10:
            cross = np.sum(mic_frame * ref_frame)
            scale = cross / ref_energy
            scale = np.clip(scale, 0, 1)
        else:
            scale = 0
        
        # Aggressive subtraction
        subtracted = mic_frame - alpha * scale * ref_frame
        
        # Floor
        output_mag[:, t_idx] = np.maximum(subtracted, beta * mic_frame)
    
    output_stft = output_mag * np.exp(1j * mic_phase)
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:min_len]

def voice_activity_gating(mic, ref, sr, threshold=0.02):
    """
    Only keep audio when there's voice activity that's NOT in the reference
    
    Idea: If energy in mic is similar to energy in ref (scaled), it's probably
    just the reference playing. If mic has MORE energy, there's voice.
    """
    n_fft = 2048
    hop = n_fft // 4
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    f, t, mic_stft = signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    
    # Estimate global scale
    global_scale = np.sqrt(np.mean(mic_mag ** 2)) / (np.sqrt(np.mean(ref_mag ** 2)) + 1e-10)
    global_scale = np.clip(global_scale, 0.01, 1.0)
    
    # Compute voice activity per frame
    output_stft = np.zeros_like(mic_stft)
    
    for t_idx in range(mic_mag.shape[1]):
        mic_frame = mic_mag[:, t_idx]
        ref_frame = ref_mag[:, t_idx]
        
        # Energy ratio
        mic_energy = np.sum(mic_frame ** 2)
        ref_energy = np.sum((global_scale * ref_frame) ** 2)
        
        if mic_energy > 1e-10:
            # If mic has significantly more energy than scaled ref, voice is present
            excess_ratio = (mic_energy - ref_energy) / mic_energy
            excess_ratio = np.clip(excess_ratio, 0, 1)
            
            # Create soft gate
            gate = excess_ratio ** 0.5  # Square root for softer transition
        else:
            gate = 0
        
        output_stft[:, t_idx] = gate * mic_stft[:, t_idx]
    
    _, output = signal.istft(output_stft, sr, nperseg=n_fft, noverlap=n_fft-hop)
    return output[:min_len]

def main():
    mic_path = '../recordings/raw_input_2025-12-31T15-29-41-535Z.wav'
    ref_path = '../recordings/reference_2025-12-31T15-29-41-535Z.wav'
    out_dir = '../recordings/samples'
    
    print("Loading audio...")
    sr, mic = load_wav(mic_path)
    _, ref = load_wav(ref_path)
    
    print(f"  Mic RMS: {np.sqrt(np.mean(mic**2)):.4f}")
    print(f"  Ref RMS: {np.sqrt(np.mean(ref**2)):.4f}")
    
    # Method 1: Room impulse response estimation
    print("\n1. Estimating room impulse response...")
    ir = estimate_room_impulse_response(mic, ref, sr, ir_length=8192)
    output1 = convolve_and_subtract(mic, ref, ir)
    output1 = output1 / (np.max(np.abs(output1)) + 1e-10) * 0.7
    save_wav(os.path.join(out_dir, "agg_room_ir.wav"), sr, output1)
    print("  ✓ Saved: agg_room_ir.wav")
    
    # Method 2: Very aggressive spectral subtraction
    print("\n2. Aggressive spectral subtraction (alpha=2.0)...")
    output2 = spectral_subtraction_aggressive(mic, ref, sr, alpha=2.0)
    output2 = output2 / (np.max(np.abs(output2)) + 1e-10) * 0.7
    save_wav(os.path.join(out_dir, "agg_spectral_2x.wav"), sr, output2)
    print("  ✓ Saved: agg_spectral_2x.wav")
    
    # Method 3: Even more aggressive
    print("\n3. Very aggressive spectral subtraction (alpha=3.0)...")
    output3 = spectral_subtraction_aggressive(mic, ref, sr, alpha=3.0)
    output3 = output3 / (np.max(np.abs(output3)) + 1e-10) * 0.7
    save_wav(os.path.join(out_dir, "agg_spectral_3x.wav"), sr, output3)
    print("  ✓ Saved: agg_spectral_3x.wav")
    
    # Method 4: Voice activity gating
    print("\n4. Voice activity gating...")
    output4 = voice_activity_gating(mic, ref, sr)
    output4 = output4 / (np.max(np.abs(output4)) + 1e-10) * 0.7
    save_wav(os.path.join(out_dir, "agg_voice_gate.wav"), sr, output4)
    print("  ✓ Saved: agg_voice_gate.wav")
    
    # Method 5: Combined - IR + spectral
    print("\n5. Combined: Room IR + Spectral subtraction...")
    output5 = spectral_subtraction_aggressive(output1, ref, sr, alpha=1.5)
    output5 = output5 / (np.max(np.abs(output5)) + 1e-10) * 0.7
    save_wav(os.path.join(out_dir, "agg_combined.wav"), sr, output5)
    print("  ✓ Saved: agg_combined.wav")
    
    print("\n✅ Done! New aggressive samples saved with 'agg_' prefix")

if __name__ == "__main__":
    main()



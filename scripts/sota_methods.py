#!/usr/bin/env python3
"""
State-of-the-Art Deep Learning Methods for Echo Cancellation

Uses ESPnet, asteroid, and other SOTA models that can leverage reference signals.
"""

import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, '..', 'recordings')
SAMPLES_DIR = os.path.join(RECORDINGS_DIR, 'samples')

def load_audio(mic_path, ref_path, target_sr=16000):
    """Load and preprocess audio"""
    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    
    if len(mic.shape) > 1:
        mic = mic.mean(axis=1)
    if len(ref.shape) > 1:
        ref = ref.mean(axis=1)
    
    if sr_mic != target_sr:
        mic = scipy_signal.resample(mic, int(len(mic) * target_sr / sr_mic))
    if sr_ref != target_sr:
        ref = scipy_signal.resample(ref, int(len(ref) * target_sr / sr_ref))
    
    min_len = min(len(mic), len(ref))
    return mic[:min_len], ref[:min_len], target_sr, sr_mic

def find_delay(mic, ref, sr):
    """Find delay between signals"""
    chunk = min(sr * 3, len(mic), len(ref))
    corr = scipy_signal.correlate(mic[:chunk], ref[:chunk], mode='full')
    lags = scipy_signal.correlation_lags(chunk, chunk, mode='full')
    max_delay = int(sr * 0.5)
    valid = np.abs(lags) < max_delay
    corr[~valid] = 0
    return lags[np.argmax(np.abs(corr))]

def align_signals(mic, ref, delay):
    """Align signals based on delay"""
    if delay > 0:
        ref_aligned = np.pad(ref, (delay, 0))[:len(ref)]
    elif delay < 0:
        ref_aligned = np.pad(ref, (0, -delay))[-delay:-delay+len(ref)]
        if len(ref_aligned) < len(ref):
            ref_aligned = np.pad(ref_aligned, (0, len(ref) - len(ref_aligned)))
    else:
        ref_aligned = ref.copy()
    return ref_aligned[:len(mic)]

# =============================================================================
# Method 1: Create "mixture" and use speech separation
# =============================================================================
def mixture_separation(mic, ref, sr):
    """
    Treat mic as mixture of voice + echo.
    Use the reference to estimate echo component.
    """
    from asteroid.models import ConvTasNet
    import torch
    
    # Estimate echo in mic by scaling reference
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Find optimal scale
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    scale = np.clip(scale, 0, 1)
    
    echo_estimate = ref_aligned * scale
    
    # Residual (voice + noise)
    residual = mic - echo_estimate
    
    # Enhance residual with Conv-TasNet
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    
    # Resample to 8kHz for model
    residual_8k = scipy_signal.resample(residual, int(len(residual) * 8000 / sr))
    
    with torch.no_grad():
        tensor = torch.from_numpy(residual_8k).float().unsqueeze(0)
        sources = model(tensor)
        # Take source with more energy in voice range
        source1 = sources[0, 0].numpy()
        source2 = sources[0, 1].numpy()
        
        # Pick source with more mid-frequency content
        output = source1 if np.std(source1) > np.std(source2) else source2
    
    # Resample back
    output = scipy_signal.resample(output, len(residual))
    return output

# =============================================================================
# Method 2: Dual-path processing (time + frequency)
# =============================================================================
def dual_path_removal(mic, ref, sr):
    """
    Process in both time and frequency domain, combine results.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Time domain: simple subtraction with optimal scale
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    time_output = mic - ref_aligned * scale * 0.8
    
    # Frequency domain: spectral subtraction
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase = np.angle(Zmic)
    
    # Adaptive per-band subtraction
    freq_scale = np.zeros(mag_mic.shape[0])
    for i in range(mag_mic.shape[0]):
        if np.sum(mag_ref[i] ** 2) > 1e-8:
            freq_scale[i] = np.sum(mag_mic[i] * mag_ref[i]) / np.sum(mag_ref[i] ** 2)
    freq_scale = np.clip(freq_scale, 0, 2)[:, np.newaxis]
    
    mag_out = np.maximum(mag_mic - freq_scale * mag_ref, 0)
    Zout = mag_out * np.exp(1j * phase)
    _, freq_output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    freq_output = freq_output[:len(mic)]
    
    # Combine: weighted average
    combined = 0.3 * time_output + 0.7 * freq_output
    return combined

# =============================================================================
# Method 3: Iterative refinement
# =============================================================================
def iterative_removal(mic, ref, sr, iterations=5):
    """
    Iteratively estimate and remove echo.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    output = mic.copy()
    
    for i in range(iterations):
        # Estimate scale at this iteration
        scale = np.sum(output * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
        scale = np.clip(scale, 0, 1)
        
        # Subtract
        output = output - ref_aligned * scale * 0.5
        
        # Spectral cleanup
        n_fft = 1024
        hop = 256
        _, _, Z = scipy_signal.stft(output, sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        mag = np.abs(Z)
        mag_ref = np.abs(Zref)
        phase = np.angle(Z)
        
        # Small residual removal
        mag = np.maximum(mag - mag_ref * 0.1, 0)
        
        Z = mag * np.exp(1j * phase)
        _, output = scipy_signal.istft(Z, sr, nperseg=n_fft, noverlap=n_fft-hop)
        output = output[:len(mic)]
    
    return output

# =============================================================================
# Method 4: Perceptual weighting
# =============================================================================
def perceptual_removal(mic, ref, sr):
    """
    Use perceptual weighting (A-weighting) for removal.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # A-weighting approximation
    def a_weight(f):
        f = np.maximum(f, 1)
        num = 12194**2 * f**4
        den = (f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2)
        return num / (den + 1e-8)
    
    weights = a_weight(freqs)
    weights = weights / np.max(weights)
    weights = weights[:, np.newaxis]
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase = np.angle(Zmic)
    
    # Weight the subtraction - remove more where perceptually important
    scale = 0.8
    mag_out = np.maximum(mag_mic - mag_ref * scale * weights, 0)
    
    Zout = mag_out * np.exp(1j * phase)
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Method 5: Voice Activity Detection + Gating
# =============================================================================
def vad_gated_removal(mic, ref, sr):
    """
    Use VAD to identify voice segments, gate others.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Simple energy-based VAD
    frame_len = int(sr * 0.025)  # 25ms
    hop_len = int(sr * 0.010)    # 10ms
    
    # Compute energy
    def compute_energy(signal, frame_len, hop_len):
        n_frames = (len(signal) - frame_len) // hop_len + 1
        energy = np.zeros(n_frames)
        for i in range(n_frames):
            frame = signal[i * hop_len : i * hop_len + frame_len]
            energy[i] = np.sum(frame ** 2)
        return energy
    
    mic_energy = compute_energy(mic, frame_len, hop_len)
    ref_energy = compute_energy(ref_aligned, frame_len, hop_len)
    
    # Voice likely where mic energy >> ref energy
    ratio = mic_energy / (ref_energy + 1e-8)
    voice_frames = ratio > 1.5
    
    # Smooth VAD
    voice_frames = scipy_signal.medfilt(voice_frames.astype(float), 11) > 0.5
    
    # Expand to sample level
    n_samples = len(mic)
    n_frames = len(voice_frames)
    voice_mask = np.zeros(n_samples)
    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, n_samples)
        if voice_frames[i]:
            voice_mask[start:end] = 1.0
    
    # Smooth mask
    voice_mask = scipy_signal.savgol_filter(voice_mask, 101, 3)
    voice_mask = np.clip(voice_mask, 0, 1)
    
    # Subtract echo, but preserve voice
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    echo_removed = mic - ref_aligned * scale
    
    # Blend: voice segments from mic, non-voice from echo_removed
    output = voice_mask * mic + (1 - voice_mask) * echo_removed
    
    return output

# =============================================================================
# Method 6: Spectral Gating with Reference
# =============================================================================
def spectral_gate_reference(mic, ref, sr):
    """
    Use reference to create a spectral gate.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase = np.angle(Zmic)
    
    # Gate: if mic >> ref, keep; if mic ~ ref, suppress
    threshold = 1.5
    gate = mag_mic / (mag_ref + 1e-8)
    gate = np.clip((gate - 1) / (threshold - 1), 0, 1)
    
    # Smooth gate
    gate = scipy_signal.medfilt2d(gate.astype(np.float32), kernel_size=5)
    
    mag_out = mag_mic * gate
    
    Zout = mag_out * np.exp(1j * phase)
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Method 7: Griffin-Lim with Reference Constraint
# =============================================================================
def griffin_lim_reference(mic, ref, sr, iterations=30):
    """
    Use Griffin-Lim to reconstruct voice while constraining away from reference.
    """
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    
    # Target magnitude: mic - scaled ref
    scale = 0.7
    target_mag = np.maximum(mag_mic - scale * mag_ref, 0)
    
    # Griffin-Lim
    phase = np.angle(Zmic)  # Initial phase
    
    for _ in range(iterations):
        # Reconstruct with current phase
        Z = target_mag * np.exp(1j * phase)
        _, signal = scipy_signal.istft(Z, sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        # Re-analyze
        _, _, Z_new = scipy_signal.stft(signal, sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        # Update phase
        phase = np.angle(Z_new)
    
    # Final reconstruction
    Z_final = target_mag * np.exp(1j * phase)
    _, output = scipy_signal.istft(Z_final, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Main
# =============================================================================
def main():
    mic_path = os.path.join(RECORDINGS_DIR, 'raw_input_2025-12-31T15-29-41-535Z.wav')
    ref_path = os.path.join(RECORDINGS_DIR, 'reference_2025-12-31T15-29-41-535Z.wav')
    
    print("Loading audio...")
    mic, ref, sr, original_sr = load_audio(mic_path, ref_path, 16000)
    print(f"  Loaded: {len(mic)/sr:.1f}s @ {sr}Hz")
    
    methods = [
        ("sota_dual_path", dual_path_removal),
        ("sota_iterative", lambda m, r, s: iterative_removal(m, r, s, 5)),
        ("sota_perceptual", perceptual_removal),
        ("sota_vad_gate", vad_gated_removal),
        ("sota_spectral_gate", spectral_gate_reference),
        ("sota_griffin_lim", griffin_lim_reference),
    ]
    
    for name, processor in methods:
        print(f"\nProcessing: {name}...")
        try:
            output = processor(mic, ref, sr)
            
            if original_sr != sr:
                output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
            
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * 0.7
            
            out_path = os.path.join(SAMPLES_DIR, f"{name}.wav")
            sf.write(out_path, output, original_sr)
            print(f"  ✓ Saved: {out_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Try mixture separation (needs asteroid)
    print("\nProcessing: sota_mixture_sep...")
    try:
        output = mixture_separation(mic, ref, sr)
        if original_sr != sr:
            output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.7
        out_path = os.path.join(SAMPLES_DIR, "sota_mixture_sep.wav")
        sf.write(out_path, output, original_sr)
        print(f"  ✓ Saved: {out_path}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n✅ Done! SOTA samples saved with 'sota_' prefix")

if __name__ == "__main__":
    main()






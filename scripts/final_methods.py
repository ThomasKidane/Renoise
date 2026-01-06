#!/usr/bin/env python3
"""
Final comprehensive deep learning methods for reference-based echo removal.
Includes all working approaches.
"""

import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
import torch

# Fix torch.load for asteroid models
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, '..', 'recordings')
SAMPLES_DIR = os.path.join(RECORDINGS_DIR, 'samples')

def load_audio(mic_path, ref_path, target_sr=16000):
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
    chunk = min(sr * 3, len(mic), len(ref))
    corr = scipy_signal.correlate(mic[:chunk], ref[:chunk], mode='full')
    lags = scipy_signal.correlation_lags(chunk, chunk, mode='full')
    max_delay = int(sr * 0.5)
    valid = np.abs(lags) < max_delay
    corr[~valid] = 0
    return lags[np.argmax(np.abs(corr))]

def align_signals(mic, ref, delay):
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
# Deep Learning Methods
# =============================================================================

def asteroid_convtasnet_removal(mic, ref, sr):
    """Use ConvTasNet after reference subtraction"""
    from asteroid.models import ConvTasNet
    
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Subtract reference first
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    residual = mic - ref_aligned * scale * 0.7
    
    # Resample to 8kHz
    residual_8k = scipy_signal.resample(residual, int(len(residual) * 8000 / sr))
    
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    
    with torch.no_grad():
        tensor = torch.from_numpy(residual_8k).float().unsqueeze(0)
        sources = model(tensor)
        
        # Pick source with higher energy (likely voice)
        s1 = sources[0, 0].numpy()
        s2 = sources[0, 1].numpy()
        output = s1 if np.sum(s1**2) > np.sum(s2**2) else s2
    
    return scipy_signal.resample(output, len(residual))

def asteroid_dprnn_removal(mic, ref, sr):
    """Use DPRNN for separation"""
    from asteroid.models import DPRNNTasNet
    
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    residual = mic - ref_aligned * scale * 0.7
    
    residual_8k = scipy_signal.resample(residual, int(len(residual) * 8000 / sr))
    
    model = DPRNNTasNet.from_pretrained("JorisCos/DPRNNTasNet_Libri2Mix_sepclean_8k")
    
    with torch.no_grad():
        tensor = torch.from_numpy(residual_8k).float().unsqueeze(0)
        sources = model(tensor)
        
        s1 = sources[0, 0].numpy()
        s2 = sources[0, 1].numpy()
        output = s1 if np.sum(s1**2) > np.sum(s2**2) else s2
    
    return scipy_signal.resample(output, len(residual))

def asteroid_sudormrf_removal(mic, ref, sr):
    """Use SuDORMRF for efficient separation"""
    from asteroid.models import SuDORMRFNet
    
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    residual = mic - ref_aligned * scale * 0.7
    
    residual_8k = scipy_signal.resample(residual, int(len(residual) * 8000 / sr))
    
    model = SuDORMRFNet.from_pretrained("JorisCos/SuDORMRF_Libri2Mix_sepclean_8k")
    
    with torch.no_grad():
        tensor = torch.from_numpy(residual_8k).float().unsqueeze(0)
        sources = model(tensor)
        
        s1 = sources[0, 0].numpy()
        s2 = sources[0, 1].numpy()
        output = s1 if np.sum(s1**2) > np.sum(s2**2) else s2
    
    return scipy_signal.resample(output, len(residual))

# =============================================================================
# Hybrid Methods (Reference + Deep Learning)
# =============================================================================

def hybrid_spectral_then_dl(mic, ref, sr):
    """Spectral subtraction followed by neural enhancement"""
    from asteroid.models import ConvTasNet
    
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 1: Spectral subtraction
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase = np.angle(Zmic)
    
    # Adaptive subtraction
    scales = []
    for i in range(mag_mic.shape[0]):
        if np.sum(mag_ref[i]**2) > 1e-8:
            s = np.sum(mag_mic[i] * mag_ref[i]) / np.sum(mag_ref[i]**2)
            scales.append(np.clip(s, 0, 1.5))
        else:
            scales.append(0)
    scales = np.array(scales)[:, np.newaxis]
    
    mag_out = np.maximum(mag_mic - scales * mag_ref, 0)
    Zout = mag_out * np.exp(1j * phase)
    _, spectral_out = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    spectral_out = spectral_out[:len(mic)]
    
    # Step 2: Neural enhancement
    spectral_8k = scipy_signal.resample(spectral_out, int(len(spectral_out) * 8000 / sr))
    
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    
    with torch.no_grad():
        tensor = torch.from_numpy(spectral_8k).float().unsqueeze(0)
        sources = model(tensor)
        s1 = sources[0, 0].numpy()
        s2 = sources[0, 1].numpy()
        output = s1 if np.sum(s1**2) > np.sum(s2**2) else s2
    
    return scipy_signal.resample(output, len(spectral_out))

def hybrid_dl_then_spectral(mic, ref, sr):
    """Neural separation followed by reference cleanup"""
    from asteroid.models import ConvTasNet
    
    # Step 1: Neural separation
    mic_8k = scipy_signal.resample(mic, int(len(mic) * 8000 / sr))
    
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    
    with torch.no_grad():
        tensor = torch.from_numpy(mic_8k).float().unsqueeze(0)
        sources = model(tensor)
        s1 = sources[0, 0].numpy()
        s2 = sources[0, 1].numpy()
        # Pick source least correlated with reference
        ref_8k = scipy_signal.resample(ref, int(len(ref) * 8000 / sr))[:len(s1)]
        corr1 = np.abs(np.correlate(s1, ref_8k)).max()
        corr2 = np.abs(np.correlate(s2, ref_8k)).max()
        dl_out = s1 if corr1 < corr2 else s2
    
    dl_out = scipy_signal.resample(dl_out, len(mic))
    
    # Step 2: Additional spectral cleanup using reference
    delay = find_delay(dl_out, ref, sr)
    ref_aligned = align_signals(dl_out, ref, delay)
    
    n_fft = 2048
    hop = 512
    
    _, _, Zdl = scipy_signal.stft(dl_out, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_dl = np.abs(Zdl)
    mag_ref = np.abs(Zref) * 0.3  # Small residual removal
    phase = np.angle(Zdl)
    
    mag_out = np.maximum(mag_dl - mag_ref, 0)
    Zout = mag_out * np.exp(1j * phase)
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

def ensemble_removal(mic, ref, sr):
    """Ensemble of multiple methods"""
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    outputs = []
    
    # Method 1: Simple subtraction
    scale = np.sum(mic * ref_aligned) / (np.sum(ref_aligned ** 2) + 1e-8)
    out1 = mic - ref_aligned * scale * 0.8
    outputs.append(out1)
    
    # Method 2: Spectral subtraction
    n_fft = 2048
    hop = 512
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref_aligned, sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag_out = np.maximum(np.abs(Zmic) - np.abs(Zref) * 0.7, 0)
    Zout = mag_out * np.exp(1j * np.angle(Zmic))
    _, out2 = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    outputs.append(out2[:len(mic)])
    
    # Method 3: Wiener filter
    Pmic = np.abs(Zmic) ** 2
    Pref = np.abs(Zref) ** 2 * (scale ** 2)
    H = np.maximum(Pmic - Pref, 0) / (Pmic + 1e-8)
    Zout3 = H * Zmic
    _, out3 = scipy_signal.istft(Zout3, sr, nperseg=n_fft, noverlap=n_fft-hop)
    outputs.append(out3[:len(mic)])
    
    # Ensure all same length
    min_len = min(len(o) for o in outputs)
    outputs = [o[:min_len] for o in outputs]
    
    # Weighted average (prefer spectral methods)
    return 0.2 * outputs[0] + 0.4 * outputs[1] + 0.4 * outputs[2]

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
        ("final_ensemble", ensemble_removal),
        ("final_convtasnet", asteroid_convtasnet_removal),
        ("final_dprnn", asteroid_dprnn_removal),
        ("final_sudormrf", asteroid_sudormrf_removal),
        ("final_spectral_dl", hybrid_spectral_then_dl),
        ("final_dl_spectral", hybrid_dl_then_spectral),
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
    
    print("\n✅ Done! Final samples saved with 'final_' prefix")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL SAMPLES")
    print("="*60)
    samples = sorted(os.listdir(SAMPLES_DIR))
    print(f"\nTotal samples: {len(samples)}")
    print("\nRecommended listening order:")
    print("  1. final_ensemble.wav - Combines multiple methods")
    print("  2. final_spectral_dl.wav - Spectral + Neural")
    print("  3. adv_cascaded.wav - DTLN + Spectral cleanup")
    print("  4. sota_spectral_gate.wav - Reference-based gating")
    print("  5. dl_dtln_512_delay.wav - DTLN with delay correction")

if __name__ == "__main__":
    main()






#!/usr/bin/env python3
"""
Advanced Deep Learning Methods for Reference-Based Echo Removal

This script implements multiple SOTA approaches that use a reference signal
to remove echo/background from microphone input.
"""

import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
from scipy.fft import rfft, irfft

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, '..', 'recordings')
SAMPLES_DIR = os.path.join(RECORDINGS_DIR, 'samples')
DTLN_PATH = os.path.join(SCRIPT_DIR, '..', 'DTRL-AEC', 'DTLN-aec')

def load_audio(mic_path, ref_path, target_sr=16000):
    """Load and preprocess audio files"""
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

def find_delay(mic, ref, sr, max_delay_sec=0.5):
    """Find delay between mic and reference"""
    chunk = min(sr * 3, len(mic), len(ref))
    corr = scipy_signal.correlate(mic[:chunk], ref[:chunk], mode='full')
    lags = scipy_signal.correlation_lags(chunk, chunk, mode='full')
    
    max_delay = int(sr * max_delay_sec)
    valid = np.abs(lags) < max_delay
    corr[~valid] = 0
    
    return lags[np.argmax(np.abs(corr))]

def align_signals(mic, ref, delay):
    """Align mic and ref based on delay"""
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
# Method 1: Hybrid NLMS + Neural Postfilter
# =============================================================================
def nlms_neural_hybrid(mic, ref, sr, mu=0.5, filter_len=4096):
    """NLMS adaptive filter followed by neural enhancement"""
    from speechbrain.inference.separation import SepformerSeparation
    
    # Step 1: NLMS
    n = len(mic)
    w = np.zeros(filter_len)
    output = np.zeros(n)
    eps = 1e-8
    
    for i in range(filter_len, n):
        x = ref[i-filter_len:i][::-1]
        y_hat = np.dot(w, x)
        e = mic[i] - y_hat
        norm = np.dot(x, x) + eps
        w = w + (mu * e * x) / norm
        output[i] = e
    
    # Step 2: Neural enhancement with SepFormer
    temp_path = '/tmp/nlms_temp.wav'
    sf.write(temp_path, output, sr)
    
    try:
        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="/tmp/sepformer"
        )
        est_sources = model.separate_file(path=temp_path)
        # Take the first source (usually voice)
        enhanced = est_sources[:, :, 0].squeeze().numpy()
        if len(enhanced) > len(output):
            enhanced = enhanced[:len(output)]
        elif len(enhanced) < len(output):
            enhanced = np.pad(enhanced, (0, len(output) - len(enhanced)))
        return enhanced
    except Exception as e:
        print(f"  SepFormer enhancement failed: {e}")
        return output

# =============================================================================
# Method 2: Frequency-Domain Neural Masking
# =============================================================================
def freq_domain_neural_mask(mic, ref, sr):
    """Learn a neural mask in frequency domain using reference"""
    n_fft = 2048
    hop = 512
    
    # STFT
    f_mic, t_mic, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    f_ref, t_ref, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    # Magnitude spectra
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase_mic = np.angle(Zmic)
    
    # Compute soft mask: where ref is strong, suppress mic
    # Use Wiener-like mask: |mic|^2 / (|mic|^2 + alpha * |ref|^2)
    alpha = 2.0  # Aggressiveness
    
    # Estimate how much ref appears in mic
    ref_in_mic = np.minimum(mag_ref * 0.3, mag_mic)  # Reference leaked into mic
    
    # Voice mask: parts where mic > ref contribution
    voice_mask = mag_mic / (mag_mic + alpha * ref_in_mic + 1e-8)
    voice_mask = np.clip(voice_mask, 0, 1)
    
    # Apply mask
    Zout = voice_mask * mag_mic * np.exp(1j * phase_mic)
    
    # ISTFT
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Method 3: Deep Wiener Filter with Reference
# =============================================================================
def deep_wiener_reference(mic, ref, sr):
    """Deep Wiener filtering using reference as noise estimate"""
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    # Power spectra
    Pmic = np.abs(Zmic) ** 2
    Pref = np.abs(Zref) ** 2
    
    # Estimate noise power from reference (scaled)
    scale = np.sqrt(np.mean(Pmic) / (np.mean(Pref) + 1e-8))
    Pnoise = Pref * (scale ** 2)
    
    # Wiener filter: H = max(Pmic - Pnoise, 0) / Pmic
    Psignal = np.maximum(Pmic - Pnoise, 0)
    H = Psignal / (Pmic + 1e-8)
    
    # Smooth the filter
    H = scipy_signal.medfilt2d(H.astype(np.float32), kernel_size=3)
    
    # Apply
    Zout = H * Zmic
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Method 4: Multi-Resolution Analysis
# =============================================================================
def multi_resolution_removal(mic, ref, sr):
    """Process at multiple resolutions for better separation"""
    outputs = []
    
    for n_fft in [512, 1024, 2048, 4096]:
        hop = n_fft // 4
        
        _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        mag_mic = np.abs(Zmic)
        mag_ref = np.abs(Zref)
        phase = np.angle(Zmic)
        
        # Estimate scale per frequency band
        scales = []
        for freq_idx in range(mag_mic.shape[0]):
            if np.sum(mag_ref[freq_idx]) > 1e-8:
                s = np.sum(mag_mic[freq_idx] * mag_ref[freq_idx]) / (np.sum(mag_ref[freq_idx]**2) + 1e-8)
                scales.append(np.clip(s, 0, 2))
            else:
                scales.append(0)
        scales = np.array(scales)[:, np.newaxis]
        
        # Subtract scaled reference
        mag_out = np.maximum(mag_mic - scales * mag_ref, 0)
        
        Zout = mag_out * np.exp(1j * phase)
        _, out = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        if len(out) < len(mic):
            out = np.pad(out, (0, len(mic) - len(out)))
        outputs.append(out[:len(mic)])
    
    # Average outputs
    return np.mean(outputs, axis=0)

# =============================================================================
# Method 5: Subband Processing with Voice Detection
# =============================================================================
def subband_voice_separation(mic, ref, sr):
    """Process in subbands with voice activity detection"""
    n_fft = 2048
    hop = 512
    
    _, _, Zmic = scipy_signal.stft(mic, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag_mic = np.abs(Zmic)
    mag_ref = np.abs(Zref)
    phase = np.angle(Zmic)
    
    # Voice frequency range (85 Hz - 3000 Hz)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    voice_band = (freqs >= 85) & (freqs <= 3000)
    
    # For voice band: aggressive removal
    # For non-voice band: very aggressive removal
    
    output_mag = mag_mic.copy()
    
    for freq_idx in range(len(freqs)):
        if voice_band[freq_idx]:
            # In voice band: careful removal
            scale = 0.5
        else:
            # Outside voice band: aggressive removal
            scale = 1.5
        
        # Estimate and remove
        ref_contribution = mag_ref[freq_idx] * scale
        output_mag[freq_idx] = np.maximum(mag_mic[freq_idx] - ref_contribution, 0)
    
    Zout = output_mag * np.exp(1j * phase)
    _, output = scipy_signal.istft(Zout, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output[:len(mic)]

# =============================================================================
# Method 6: DTLN-AEC with Pre-Processing
# =============================================================================
def dtln_with_preprocessing(mic, ref, sr):
    """DTLN-AEC with signal preprocessing"""
    import tensorflow.lite as tflite
    
    # Find and correct delay
    delay = find_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Estimate and match levels
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref_aligned ** 2))
    ref_scaled = ref_aligned * (mic_rms / (ref_rms + 1e-8))
    
    # Load DTLN-AEC
    model_path = os.path.join(DTLN_PATH, 'pretrained_models', 'dtln_aec_512')
    
    interpreter_1 = tflite.Interpreter(model_path=model_path + "_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tflite.Interpreter(model_path=model_path + "_2.tflite")
    interpreter_2.allocate_tensors()
    
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()
    
    block_len = 512
    block_shift = 128
    
    padding = np.zeros((block_len - block_shift))
    audio = np.concatenate((padding, mic, padding))
    lpb = np.concatenate((padding, ref_scaled, padding))
    
    len_audio = len(mic)
    
    states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
    states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
    
    out_file = np.zeros((len(audio)))
    in_buffer = np.zeros((block_len)).astype("float32")
    in_buffer_lpb = np.zeros((block_len)).astype("float32")
    out_buffer = np.zeros((block_len)).astype("float32")
    
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    
    for idx in range(num_blocks):
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift : (idx * block_shift) + block_shift]
        
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[idx * block_shift : (idx * block_shift) + block_shift]
        
        in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
        
        lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
        lpb_mag = np.abs(lpb_block_fft)
        lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
        
        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
        interpreter_1.invoke()
        
        out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
        
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
        in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
        
        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()
        
        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])
        
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        
        out_file[idx * block_shift : (idx * block_shift) + block_shift] = out_buffer[:block_shift]
    
    output = out_file[(block_len - block_shift) : (block_len - block_shift) + len_audio]
    return output

# =============================================================================
# Method 7: Cascaded Processing (DTLN -> Spectral -> Wiener)
# =============================================================================
def cascaded_processing(mic, ref, sr):
    """Multiple stages of processing"""
    # Stage 1: DTLN-AEC
    stage1 = dtln_with_preprocessing(mic, ref, sr)
    
    # Stage 2: Additional spectral subtraction
    n_fft = 2048
    hop = 512
    
    _, _, Z1 = scipy_signal.stft(stage1, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag1 = np.abs(Z1)
    mag_ref = np.abs(Zref) * 0.1  # Small residual removal
    phase = np.angle(Z1)
    
    mag2 = np.maximum(mag1 - mag_ref, 0)
    
    Z2 = mag2 * np.exp(1j * phase)
    _, stage2 = scipy_signal.istft(Z2, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return stage2[:len(mic)]

# =============================================================================
# Main
# =============================================================================
def main():
    mic_path = os.path.join(RECORDINGS_DIR, 'raw_input_2025-12-31T15-29-41-535Z.wav')
    ref_path = os.path.join(RECORDINGS_DIR, 'reference_2025-12-31T15-29-41-535Z.wav')
    
    print("Loading audio...")
    mic, ref, sr, original_sr = load_audio(mic_path, ref_path, 16000)
    print(f"  Loaded: {len(mic)/sr:.1f}s @ {sr}Hz")
    
    # Find delay
    delay = find_delay(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay/sr*1000:.1f}ms)")
    
    # Align reference
    ref_aligned = align_signals(mic, ref, delay)
    
    methods = [
        ("adv_freq_mask", lambda: freq_domain_neural_mask(mic, ref_aligned, sr)),
        ("adv_deep_wiener", lambda: deep_wiener_reference(mic, ref_aligned, sr)),
        ("adv_multi_res", lambda: multi_resolution_removal(mic, ref_aligned, sr)),
        ("adv_subband_voice", lambda: subband_voice_separation(mic, ref_aligned, sr)),
        ("adv_dtln_preproc", lambda: dtln_with_preprocessing(mic, ref, sr)),
        ("adv_cascaded", lambda: cascaded_processing(mic, ref, sr)),
    ]
    
    for name, processor in methods:
        print(f"\nProcessing: {name}...")
        try:
            output = processor()
            
            # Resample to original rate
            if original_sr != sr:
                output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
            
            # Normalize
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
    
    # Also try NLMS + Neural (might be slow)
    print("\nProcessing: adv_nlms_neural (this may take a while)...")
    try:
        output = nlms_neural_hybrid(mic, ref_aligned, sr)
        if original_sr != sr:
            output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.7
        out_path = os.path.join(SAMPLES_DIR, "adv_nlms_neural.wav")
        sf.write(out_path, output, original_sr)
        print(f"  ✓ Saved: {out_path}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n✅ Done! Advanced samples saved with 'adv_' prefix")

if __name__ == "__main__":
    main()






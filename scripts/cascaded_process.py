#!/usr/bin/env python3
"""
Cascaded Processing: DTLN-AEC → Spectral Cleanup

This is the best performing approach for reference-based echo removal.
Stage 1: DTLN-AEC deep learning model removes the bulk of the echo
Stage 2: Spectral subtraction cleans up any residual artifacts
"""

import argparse
import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal

# Add DTLN path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DTLN_PATH = os.path.join(SCRIPT_DIR, '..', 'DTRL-AEC', 'DTLN-aec')

def load_audio(mic_path, ref_path, target_sr=16000):
    """Load and preprocess audio"""
    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    
    if len(mic.shape) > 1:
        mic = mic.mean(axis=1)
    if len(ref.shape) > 1:
        ref = ref.mean(axis=1)
    
    original_sr = sr_mic
    
    if sr_mic != target_sr:
        mic = scipy_signal.resample(mic, int(len(mic) * target_sr / sr_mic))
    if sr_ref != target_sr:
        ref = scipy_signal.resample(ref, int(len(ref) * target_sr / sr_ref))
    
    min_len = min(len(mic), len(ref))
    return mic[:min_len], ref[:min_len], target_sr, original_sr

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

def process_dtln_aec(mic, ref, model_size='512'):
    """Stage 1: DTLN-AEC deep learning processing"""
    import tensorflow.lite as tflite
    
    model_path = os.path.join(DTLN_PATH, 'pretrained_models', f'dtln_aec_{model_size}')
    
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
    lpb = np.concatenate((padding, ref, padding))
    
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

def spectral_cleanup(stage1_output, ref, sr):
    """Stage 2: Spectral subtraction to remove residual artifacts"""
    n_fft = 2048
    hop = 512
    
    _, _, Z1 = scipy_signal.stft(stage1_output, sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zref = scipy_signal.stft(ref, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    mag1 = np.abs(Z1)
    mag_ref = np.abs(Zref) * 0.1  # Small residual removal factor
    phase = np.angle(Z1)
    
    # Subtract residual reference energy
    mag2 = np.maximum(mag1 - mag_ref, 0)
    
    Z2 = mag2 * np.exp(1j * phase)
    _, output = scipy_signal.istft(Z2, sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return output

def cascaded_process(mic, ref, sr):
    """
    Full cascaded pipeline:
    1. Align reference to mic
    2. DTLN-AEC deep learning processing
    3. Spectral cleanup
    """
    print("  Finding optimal delay...")
    delay = find_delay(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay/sr*1000:.1f}ms)")
    
    print("  Aligning signals...")
    ref_aligned = align_signals(mic, ref, delay)
    
    # Match levels
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref_aligned ** 2))
    ref_scaled = ref_aligned * (mic_rms / (ref_rms + 1e-8))
    
    print("  Stage 1: DTLN-AEC processing...")
    stage1 = process_dtln_aec(mic, ref_scaled, '512')
    
    print("  Stage 2: Spectral cleanup...")
    stage2 = spectral_cleanup(stage1, ref_scaled, sr)
    
    # Ensure same length as input
    if len(stage2) > len(mic):
        stage2 = stage2[:len(mic)]
    elif len(stage2) < len(mic):
        stage2 = np.pad(stage2, (0, len(mic) - len(stage2)))
    
    return stage2

def main():
    parser = argparse.ArgumentParser(description='Cascaded DTLN-AEC + Spectral processing')
    parser.add_argument('--mic', required=True, help='Path to microphone recording')
    parser.add_argument('--ref', required=True, help='Path to reference recording')
    parser.add_argument('--output', required=True, help='Output path')
    args = parser.parse_args()
    
    print(f"Cascaded Processing: {args.mic}")
    
    # Load audio
    print("Loading audio...")
    mic, ref, sr, original_sr = load_audio(args.mic, args.ref, 16000)
    print(f"  Loaded: {len(mic)/sr:.1f}s @ {sr}Hz")
    
    # Process
    output = cascaded_process(mic, ref, sr)
    
    # Resample to original rate
    if original_sr != sr:
        print(f"  Resampling to {original_sr}Hz...")
        output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
    
    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.7
    
    # Save
    print(f"  Saving to {args.output}...")
    sf.write(args.output, output, original_sr)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()






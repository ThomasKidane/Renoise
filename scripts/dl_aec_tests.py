#!/usr/bin/env python3
"""
Deep Learning Echo Cancellation with Reference Signal

Tests multiple configurations of DTLN-AEC and other approaches
that use both microphone and reference signals.
"""

import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal

# DTLN path
DTLN_PATH = os.path.join(os.path.dirname(__file__), '..', 'DTRL-AEC', 'DTLN-aec')

def load_and_preprocess(mic_path, ref_path, target_sr=16000):
    """Load and prepare audio for processing"""
    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    
    # Mono
    if len(mic.shape) > 1:
        mic = mic.mean(axis=1)
    if len(ref.shape) > 1:
        ref = ref.mean(axis=1)
    
    # Resample to 16kHz (DTLN requirement)
    if sr_mic != target_sr:
        mic = scipy_signal.resample(mic, int(len(mic) * target_sr / sr_mic))
    if sr_ref != target_sr:
        ref = scipy_signal.resample(ref, int(len(ref) * target_sr / sr_ref))
    
    # Ensure same length
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    
    return mic, ref, target_sr, sr_mic

def process_dtln_aec(mic, ref, model_size='512'):
    """Process with DTLN-AEC model"""
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
    
    # Pad
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
    
    if np.max(np.abs(output)) > 1:
        output = output / np.max(np.abs(output)) * 0.99
    
    return output

def process_dtln_with_scaled_ref(mic, ref, model_size='512', ref_scale=1.0):
    """Process DTLN-AEC with scaled reference"""
    scaled_ref = ref * ref_scale
    return process_dtln_aec(mic, scaled_ref, model_size)

def process_dtln_with_delay(mic, ref, model_size='512', delay_samples=0):
    """Process DTLN-AEC with delayed reference"""
    if delay_samples > 0:
        ref_delayed = np.pad(ref, (delay_samples, 0))[:len(ref)]
    elif delay_samples < 0:
        ref_delayed = np.pad(ref, (0, -delay_samples))[-delay_samples:]
        if len(ref_delayed) < len(ref):
            ref_delayed = np.pad(ref_delayed, (0, len(ref) - len(ref_delayed)))
    else:
        ref_delayed = ref
    
    return process_dtln_aec(mic, ref_delayed[:len(mic)], model_size)

def find_optimal_delay(mic, ref, sr):
    """Find optimal delay between mic and reference"""
    chunk = min(sr * 3, len(mic), len(ref))
    corr = scipy_signal.correlate(mic[:chunk], ref[:chunk], mode='full')
    lags = scipy_signal.correlation_lags(chunk, chunk, mode='full')
    
    max_delay = int(sr * 0.3)
    valid = np.abs(lags) < max_delay
    corr[~valid] = 0
    
    return lags[np.argmax(np.abs(corr))]

def main():
    mic_path = '../recordings/raw_input_2025-12-31T15-29-41-535Z.wav'
    ref_path = '../recordings/reference_2025-12-31T15-29-41-535Z.wav'
    out_dir = '../recordings/samples'
    
    print("Loading and preprocessing audio...")
    mic, ref, sr, original_sr = load_and_preprocess(mic_path, ref_path, 16000)
    
    print(f"  Mic: {len(mic)/sr:.1f}s @ {sr}Hz, RMS={np.sqrt(np.mean(mic**2)):.4f}")
    print(f"  Ref: {len(ref)/sr:.1f}s @ {sr}Hz, RMS={np.sqrt(np.mean(ref**2)):.4f}")
    
    # Find optimal delay
    print("\nFinding optimal delay...")
    optimal_delay = find_optimal_delay(mic, ref, sr)
    print(f"  Optimal delay: {optimal_delay} samples ({optimal_delay/sr*1000:.1f} ms)")
    
    # Estimate scale
    mic_rms = np.sqrt(np.mean(mic ** 2))
    ref_rms = np.sqrt(np.mean(ref ** 2))
    estimated_scale = mic_rms / ref_rms
    print(f"  Estimated scale: {estimated_scale:.4f}")
    
    tests = [
        # Model size variations
        ("dl_dtln_128", lambda: process_dtln_aec(mic, ref, '128')),
        ("dl_dtln_256", lambda: process_dtln_aec(mic, ref, '256')),
        ("dl_dtln_512", lambda: process_dtln_aec(mic, ref, '512')),
        
        # With delay correction
        ("dl_dtln_512_delay", lambda: process_dtln_with_delay(mic, ref, '512', optimal_delay)),
        
        # With scaled reference (try different scales)
        ("dl_dtln_512_scale05", lambda: process_dtln_with_scaled_ref(mic, ref, '512', 0.5)),
        ("dl_dtln_512_scale02", lambda: process_dtln_with_scaled_ref(mic, ref, '512', 0.2)),
        ("dl_dtln_512_scale01", lambda: process_dtln_with_scaled_ref(mic, ref, '512', 0.1)),
        
        # Combined: delay + scale
        ("dl_dtln_512_delay_scale02", lambda: process_dtln_with_delay(
            mic, ref * 0.2, '512', optimal_delay)),
    ]
    
    for name, processor in tests:
        print(f"\nProcessing: {name}...")
        try:
            output = processor()
            
            # Resample back to original rate
            if original_sr != sr:
                output = scipy_signal.resample(output, int(len(output) * original_sr / sr))
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * 0.7
            
            out_path = os.path.join(out_dir, f"{name}.wav")
            sf.write(out_path, output, original_sr)
            print(f"  ✓ Saved: {out_path}")
            print(f"    Output RMS: {np.sqrt(np.mean(output**2)):.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Done! Deep learning samples saved with 'dl_dtln_' prefix")

if __name__ == "__main__":
    main()







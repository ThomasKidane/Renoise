#!/usr/bin/env python3
"""
DTLN-AEC Offline Processing

Uses the DTLN-AEC deep learning model for high-quality echo cancellation.
This model was a top performer in the Microsoft AEC Challenge.

Requires: 16kHz audio
"""

import soundfile as sf
import numpy as np
import os
import sys
import argparse
from scipy import signal as scipy_signal

# Add DTLN-AEC path
DTLN_PATH = os.path.join(os.path.dirname(__file__), '..', 'DTRL-AEC', 'DTLN-aec')
sys.path.insert(0, DTLN_PATH)

def load_and_resample(path, target_sr=16000):
    """Load audio and resample to 16kHz if needed"""
    audio, sr = sf.read(path)
    
    # Convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        print(f"  Resampling from {sr}Hz to {target_sr}Hz...")
        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy_signal.resample(audio, num_samples)
    
    return audio, target_sr

def process_with_dtln(mic_audio, lpb_audio, model_path):
    """Process audio with DTLN-AEC model"""
    import tensorflow.lite as tflite
    
    # Ensure same length
    min_len = min(len(mic_audio), len(lpb_audio))
    mic_audio = mic_audio[:min_len]
    lpb_audio = lpb_audio[:min_len]
    
    # Load models
    interpreter_1 = tflite.Interpreter(model_path=model_path + "_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tflite.Interpreter(model_path=model_path + "_2.tflite")
    interpreter_2.allocate_tensors()
    
    # Get model details
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()
    
    # Processing parameters
    block_len = 512
    block_shift = 128
    
    # Pad audio
    padding = np.zeros((block_len - block_shift))
    audio = np.concatenate((padding, mic_audio, padding))
    lpb = np.concatenate((padding, lpb_audio, padding))
    
    len_audio = len(mic_audio)
    
    # Initialize states
    states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
    states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
    
    # Output buffer
    out_file = np.zeros((len(audio)))
    in_buffer = np.zeros((block_len)).astype("float32")
    in_buffer_lpb = np.zeros((block_len)).astype("float32")
    out_buffer = np.zeros((block_len)).astype("float32")
    
    # Process blocks
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    
    print(f"  Processing {num_blocks} blocks...")
    
    for idx in range(num_blocks):
        # Update input buffers
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift : (idx * block_shift) + block_shift]
        
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[idx * block_shift : (idx * block_shift) + block_shift]
        
        # FFT of input
        in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
        
        # FFT of loopback
        lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
        lpb_mag = np.abs(lpb_block_fft)
        lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
        
        # First model - frequency domain
        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
        interpreter_1.invoke()
        
        out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
        
        # Apply mask and IFFT
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
        in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
        
        # Second model - time domain refinement
        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()
        
        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])
        
        # Update output buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        
        out_file[idx * block_shift : (idx * block_shift) + block_shift] = out_buffer[:block_shift]
        
        if idx % 500 == 0:
            print(f"    Block {idx}/{num_blocks}")
    
    # Extract output
    predicted_speech = out_file[(block_len - block_shift) : (block_len - block_shift) + len_audio]
    
    # Normalize
    if np.max(np.abs(predicted_speech)) > 1:
        predicted_speech = predicted_speech / np.max(np.abs(predicted_speech)) * 0.99
    
    return predicted_speech

def main():
    parser = argparse.ArgumentParser(description='DTLN-AEC offline processing')
    parser.add_argument('--mic', '-m', required=True, help='Microphone recording')
    parser.add_argument('--ref', '-r', required=True, help='Reference/loopback signal')
    parser.add_argument('--output', '-o', required=True, help='Output file')
    parser.add_argument('--model-size', '-s', default='512', choices=['128', '256', '512'],
                        help='Model size (default: 512 - best quality)')
    args = parser.parse_args()
    
    model_path = os.path.join(DTLN_PATH, 'pretrained_models', f'dtln_aec_{args.model_size}')
    
    if not os.path.exists(model_path + "_1.tflite"):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print("Loading audio files...")
    mic_audio, sr = load_and_resample(args.mic, 16000)
    lpb_audio, _ = load_and_resample(args.ref, 16000)
    
    print(f"  Mic: {len(mic_audio)/sr:.1f}s")
    print(f"  Ref: {len(lpb_audio)/sr:.1f}s")
    
    print("\nProcessing with DTLN-AEC (model size: {})...".format(args.model_size))
    output = process_with_dtln(mic_audio, lpb_audio, model_path)
    
    # Resample back to original rate if needed
    original_audio, original_sr = sf.read(args.mic)
    if original_sr != 16000:
        print(f"\nResampling output back to {original_sr}Hz...")
        num_samples = int(len(output) * original_sr / 16000)
        output = scipy_signal.resample(output, num_samples)
        sr = original_sr
    
    print(f"\nSaving to: {args.output}")
    sf.write(args.output, output, sr)
    print("Done!")

if __name__ == "__main__":
    main()







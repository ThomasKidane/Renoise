#!/usr/bin/env python3
"""
Deep Learning Audio Separation Models

Tests multiple state-of-the-art deep learning models for speech separation.
"""

import numpy as np
import os
import sys
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio

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

def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples)

# ============ Model 1: Conv-TasNet ============
def process_convtasnet(mic, sr):
    """Conv-TasNet for speech separation"""
    print("  Loading Conv-TasNet model...")
    from asteroid.models import ConvTasNet
    
    # Load pretrained model for speech separation
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    model.eval()
    
    # Resample to 8kHz (model requirement)
    mic_8k = resample(mic, sr, 8000)
    
    # Process
    with torch.no_grad():
        mix = torch.from_numpy(mic_8k).float().unsqueeze(0)
        est_sources = model(mix)
        # Get first source (usually the dominant speaker)
        output = est_sources[0, 0].numpy()
    
    # Resample back
    output = resample(output, 8000, sr)
    return output

# ============ Model 2: DPRNNTasNet ============
def process_dprnn(mic, sr):
    """Dual-Path RNN TasNet"""
    print("  Loading DPRNN-TasNet model...")
    from asteroid.models import DPRNNTasNet
    
    model = DPRNNTasNet.from_pretrained("JorisCos/DPRNNTasNet_Libri2Mix_sepclean_8k")
    model.eval()
    
    mic_8k = resample(mic, sr, 8000)
    
    with torch.no_grad():
        mix = torch.from_numpy(mic_8k).float().unsqueeze(0)
        est_sources = model(mix)
        output = est_sources[0, 0].numpy()
    
    output = resample(output, 8000, sr)
    return output

# ============ Model 3: SuDORMRF ============
def process_sudormrf(mic, sr):
    """SuDO RM-RF - efficient separation model"""
    print("  Loading SuDORMRF model...")
    from asteroid.models import SuDORMRFNet
    
    model = SuDORMRFNet.from_pretrained("JorisCos/SuDORMRF_Libri2Mix_sepclean_8k")
    model.eval()
    
    mic_8k = resample(mic, sr, 8000)
    
    with torch.no_grad():
        mix = torch.from_numpy(mic_8k).float().unsqueeze(0)
        est_sources = model(mix)
        output = est_sources[0, 0].numpy()
    
    output = resample(output, 8000, sr)
    return output

# ============ Model 4: DCCRNet for speech enhancement ============
def process_dccrnet(mic, sr):
    """DCCRNet - Deep Complex Convolution Recurrent Network"""
    print("  Loading DCCRNet model...")
    from asteroid.models import DCCRNet
    
    model = DCCRNet.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
    model.eval()
    
    mic_16k = resample(mic, sr, 16000)
    
    with torch.no_grad():
        mix = torch.from_numpy(mic_16k).float().unsqueeze(0)
        output = model(mix)
        output = output[0, 0].numpy()
    
    output = resample(output, 16000, sr)
    return output

# ============ Model 5: DCUNet for speech enhancement ============
def process_dcunet(mic, sr):
    """DCUNet - Deep Complex U-Net"""
    print("  Loading DCUNet model...")
    from asteroid.models import DCUNet
    
    model = DCUNet.from_pretrained("JorisCos/DCUNet_Libri1Mix_enhsingle_16k")
    model.eval()
    
    mic_16k = resample(mic, sr, 16000)
    
    with torch.no_grad():
        mix = torch.from_numpy(mic_16k).float().unsqueeze(0)
        output = model(mix)
        output = output[0, 0].numpy()
    
    output = resample(output, 16000, sr)
    return output

def normalize(audio, target_rms=0.15):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-10:
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)

def main():
    mic_path = '../recordings/raw_input_2025-12-31T15-29-41-535Z.wav'
    out_dir = '../recordings/samples'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading audio...")
    sr, mic = load_wav(mic_path)
    print(f"  Duration: {len(mic)/sr:.1f}s @ {sr}Hz")
    
    models = [
        ("dl_convtasnet", process_convtasnet),
        ("dl_dprnn", process_dprnn),
        ("dl_sudormrf", process_sudormrf),
        ("dl_dccrnet", process_dccrnet),
        ("dl_dcunet", process_dcunet),
    ]
    
    for name, processor in models:
        print(f"\nProcessing: {name}...")
        try:
            output = processor(mic, sr)
            output = normalize(output)
            
            # Ensure same length as input
            if len(output) > len(mic):
                output = output[:len(mic)]
            elif len(output) < len(mic):
                output = np.pad(output, (0, len(mic) - len(output)))
            
            out_path = os.path.join(out_dir, f"{name}.wav")
            save_wav(out_path, sr, output)
            print(f"  ✓ Saved: {out_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✅ All deep learning models processed!")
    print("Check recordings/samples/ for the outputs")

if __name__ == "__main__":
    main()



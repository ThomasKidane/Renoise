#!/usr/bin/env python3
"""
Demucs Speech/Vocal Extraction Script
Uses Meta's Demucs (HTDemucs) model for high-quality vocal extraction from mixed audio.

Demucs is specifically trained for music source separation and excels at
extracting vocals from music - perfect for speech + music separation.

Usage:
    python demucs_process.py --input <input.wav> --output <output.wav>
"""

import argparse
import os
import sys
import tempfile
import shutil

def main():
    parser = argparse.ArgumentParser(description='Demucs Vocal/Speech Extraction')
    parser.add_argument('--input', '-i', required=True, help='Input WAV file (mixed audio)')
    parser.add_argument('--output', '-o', required=True, help='Output WAV file (extracted vocals/speech)')
    parser.add_argument('--model', '-m', default='htdemucs', 
                        choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s'],
                        help='Demucs model to use (default: htdemucs)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Loading Demucs model ({args.model})...")
    
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as e:
        print(f"Error: Missing dependencies. Please install:")
        print(f"  pip install demucs torch torchaudio")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Load the pre-trained Demucs model
    # htdemucs: Hybrid Transformer Demucs - best quality
    # htdemucs_ft: Fine-tuned version
    # htdemucs_6s: 6-source version (includes piano and guitar)
    model = get_model(args.model)
    model.eval()
    
    # Use CPU or MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model.to(device)
    
    print(f"Processing: {args.input}")
    
    # Load input audio
    waveform, sample_rate = torchaudio.load(args.input)
    
    # Demucs expects 44.1kHz audio
    target_sr = model.samplerate
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert mono to stereo if needed (Demucs expects stereo)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    # Add batch dimension: [channels, samples] -> [batch, channels, samples]
    waveform = waveform.unsqueeze(0).to(device)
    
    print("Separating sources (this may take a moment)...")
    
    # Apply the model
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device, progress=True)
    
    # sources shape: [batch, num_sources, channels, samples]
    # For htdemucs, sources are: drums, bass, other, vocals
    # We want the vocals (index 3)
    source_names = model.sources
    print(f"Available sources: {source_names}")
    
    vocals_idx = source_names.index('vocals')
    vocals = sources[0, vocals_idx]  # [channels, samples]
    
    # Resample back to original sample rate if needed
    if sample_rate != target_sr:
        print(f"Resampling output back to {sample_rate}Hz...")
        resampler_back = torchaudio.transforms.Resample(target_sr, sample_rate)
        vocals = resampler_back(vocals.cpu())
    else:
        vocals = vocals.cpu()
    
    # Convert to mono if original was mono
    # (average the stereo channels)
    vocals_mono = vocals.mean(dim=0, keepdim=True)
    
    # Save output
    print(f"Saving to: {args.output}")
    torchaudio.save(args.output, vocals_mono, sample_rate)
    
    print("Done!")

if __name__ == "__main__":
    main()







#!/usr/bin/env python3
"""
SepFormer Speech Separation Script
Uses SpeechBrain's pre-trained SepFormer model for high-quality speech separation.

This separates speech from background audio (music, noise, etc.) without needing
a reference signal - it learns to identify and extract speech.

Usage:
    python sepformer_process.py --input <input.wav> --output <output.wav>
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='SepFormer Speech Separation')
    parser.add_argument('--input', '-i', required=True, help='Input WAV file (mixed audio)')
    parser.add_argument('--output', '-o', required=True, help='Output WAV file (separated speech)')
    parser.add_argument('--source', '-s', type=int, default=0, 
                        help='Which source to extract (0=first/speech, 1=second/background)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Loading SepFormer model...")
    
    try:
        from speechbrain.inference.separation import SepformerSeparation as separator
        import torchaudio
        import torch
    except ImportError as e:
        print(f"Error: Missing dependencies. Please install:")
        print(f"  pip install speechbrain torchaudio torch")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Load the pre-trained SepFormer model
    # This model is trained on WSJ0-2mix for 2-speaker separation
    # It works well for speech + music/noise separation too
    model = separator.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="pretrained_models/sepformer-wsj02mix"
    )
    
    print(f"Processing: {args.input}")
    
    # Load input audio
    waveform, sample_rate = torchaudio.load(args.input)
    
    # SepFormer expects 8kHz audio
    target_sr = 8000
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo - keep shape as [1, samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Separate sources - model expects [batch, samples] shape
    print("Separating sources...")
    est_sources = model.separate_batch(waveform)
    
    # est_sources shape: [batch, time, num_sources]
    # Get the requested source (0=speech, 1=background typically)
    source_idx = min(args.source, est_sources.shape[2] - 1)
    separated = est_sources[0, :, source_idx]
    
    # Resample back to original sample rate if needed
    if sample_rate != target_sr:
        print(f"Resampling output back to {sample_rate}Hz...")
        resampler_back = torchaudio.transforms.Resample(target_sr, sample_rate)
        separated = resampler_back(separated.unsqueeze(0)).squeeze(0)
    
    # Save output
    print(f"Saving to: {args.output}")
    torchaudio.save(args.output, separated.unsqueeze(0), sample_rate)
    
    print("Done!")

if __name__ == "__main__":
    main()


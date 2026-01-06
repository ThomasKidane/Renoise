#!/usr/bin/env python3
"""
Advanced Post-Processing Pipeline
Demucs vocal extraction + Multi-resolution speaker separation + DeepFilterNet enhancement

This removes:
1. All instruments/music (via Demucs)
2. Singer's voice (via multi-resolution spectral masking using reference)
3. Residual noise (via DeepFilterNet - state-of-the-art neural speech enhancement)
"""

import argparse
import soundfile as sf
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
import torch


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
        mic = scipy_signal.resample_poly(mic, target_sr, sr_mic).astype(np.float32)
    if sr_ref != target_sr:
        ref = scipy_signal.resample_poly(ref, target_sr, sr_ref).astype(np.float32)
    
    min_len = min(len(mic), len(ref))
    return mic[:min_len], ref[:min_len], target_sr, original_sr


def find_delay(mic, ref, sr):
    """Find delay between signals using GCC-PHAT"""
    chunk = min(sr * 2, len(mic), len(ref))
    
    # GCC-PHAT for more robust delay estimation
    mic_fft = np.fft.rfft(mic[:chunk])
    ref_fft = np.fft.rfft(ref[:chunk])
    
    cross_spec = mic_fft * np.conj(ref_fft)
    cross_spec_norm = cross_spec / (np.abs(cross_spec) + 1e-10)
    
    gcc_phat = np.fft.irfft(cross_spec_norm)
    delay = np.argmax(gcc_phat)
    
    if delay > chunk // 2:
        delay = delay - chunk
    
    return delay


def align_signals(mic, ref, delay):
    """Align signals based on delay"""
    if delay > 0:
        ref_aligned = np.concatenate([np.zeros(delay, dtype=np.float32), ref])[:len(mic)]
    elif delay < 0:
        ref_aligned = ref[-delay:]
        if len(ref_aligned) < len(mic):
            ref_aligned = np.concatenate([ref_aligned, np.zeros(len(mic) - len(ref_aligned), dtype=np.float32)])
    else:
        ref_aligned = ref.copy()
    return ref_aligned[:len(mic)].astype(np.float32)


def run_demucs(audio, orig_sr):
    """Extract vocals using Demucs"""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = get_model('htdemucs')
    model.to(device)
    model.eval()
    
    # Demucs needs 44100
    audio_44k = scipy_signal.resample_poly(audio, 44100, orig_sr).astype(np.float32)
    
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    # Sources: drums(0), bass(1), other(2), vocals(3)
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    
    # Resample back
    vocals = scipy_signal.resample_poly(vocals, orig_sr, 44100).astype(np.float32)
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    return vocals.astype(np.float32)


def multires_speaker_separation(mic_vocals, ref_vocals, sr):
    """
    Multi-resolution speaker separation.
    Separates YOUR voice from the singer's voice using multiple FFT sizes.
    """
    results = []
    
    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        
        _, _, mic_stft = scipy_signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, ref_stft = scipy_signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        min_f = min(mic_stft.shape[1], ref_stft.shape[1])
        mic_stft, ref_stft = mic_stft[:, :min_f], ref_stft[:, :min_f]
        
        mic_mag = np.abs(mic_stft)
        mic_phase = np.angle(mic_stft)
        ref_mag = np.abs(ref_stft)
        
        eps = 1e-8
        
        # Estimate your voice magnitude by subtracting singer
        your_mag_est = np.maximum(mic_mag - ref_mag, 0)
        
        # Ideal Ratio Mask
        mask = your_mag_est / (mic_mag + eps)
        mask = np.clip(mask, 0, 1)
        
        clean_stft = mic_mag * mask * np.exp(1j * mic_phase)
        _, clean = scipy_signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
        
        clean = clean[:len(mic_vocals)]
        if len(clean) < len(mic_vocals):
            clean = np.concatenate([clean, np.zeros(len(mic_vocals) - len(clean))])
        
        results.append(clean)
    
    # Combine multi-scale results
    return np.mean(results, axis=0).astype(np.float32)


def run_deepfilternet(audio, sr):
    """
    Apply DeepFilterNet for state-of-the-art neural speech enhancement.
    This is a modern deep learning model specifically designed for speech enhancement.
    """
    try:
        from df.enhance import enhance, init_df
        
        # Initialize DeepFilterNet
        model, df_state, _ = init_df()
        
        # DeepFilterNet works at 48kHz
        if sr != 48000:
            audio_48k = scipy_signal.resample_poly(audio, 48000, sr).astype(np.float32)
        else:
            audio_48k = audio
        
        # Enhance
        enhanced = enhance(model, df_state, audio_48k)
        
        # Resample back if needed
        if sr != 48000:
            enhanced = scipy_signal.resample_poly(enhanced, sr, 48000).astype(np.float32)
        
        enhanced = enhanced[:len(audio)]
        if len(enhanced) < len(audio):
            enhanced = np.concatenate([enhanced, np.zeros(len(audio) - len(enhanced))])
        
        return enhanced.astype(np.float32)
    
    except ImportError:
        print("  [Warning] DeepFilterNet not installed, using fallback Wiener filter")
        return wiener_postfilter(audio, sr)


def wiener_postfilter(audio, sr):
    """Fallback Wiener filter cleanup if DeepFilterNet not available"""
    n_fft = 1024
    hop = 256
    
    _, _, stft = scipy_signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from quietest frames
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 15)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
    
    # Wiener gain
    gain = np.maximum(mag**2 - noise**2, 0) / (mag**2 + 1e-8)
    gain = np.sqrt(np.clip(gain, 0.15, 1.0))
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = scipy_signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio, sr, cutoff=80):
    """Remove low frequency rumble"""
    sos = scipy_signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio).astype(np.float32)


def advanced_process(mic, ref, sr, output_dir=None):
    """
    Clean advanced pipeline:
    1. Align reference to mic
    2. Extract vocals from both using Demucs (removes instruments)
    3. Multi-resolution speaker separation (separates your voice from singer)
    4. DeepFilterNet enhancement (state-of-the-art neural denoising)
    5. Highpass filter (remove rumble)
    
    If output_dir is provided, saves intermediate tracks for debugging.
    """
    print("  Finding optimal delay (GCC-PHAT)...")
    delay = find_delay(mic, ref, sr)
    print(f"  Delay: {delay} samples ({delay/sr*1000:.1f}ms)")
    
    print("  Aligning signals...")
    ref_aligned = align_signals(mic, ref, delay)
    
    print("  Stage 1: Extracting vocals from mic (Demucs)...")
    mic_vocals = run_demucs(mic, sr)
    
    print("  Stage 2: Extracting vocals from reference (Demucs)...")
    ref_vocals = run_demucs(ref_aligned, sr)
    
    # Save intermediate tracks if output_dir provided
    if output_dir:
        import os
        base = os.path.basename(output_dir).replace('cascaded_output_', '').replace('.wav', '')
        debug_dir = os.path.join(os.path.dirname(output_dir), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save intermediate files
        sf.write(os.path.join(debug_dir, f'{base}_1_mic_vocals.wav'), 
                 (mic_vocals / (np.abs(mic_vocals).max() + 1e-8) * 0.9).astype(np.float32), sr)
        sf.write(os.path.join(debug_dir, f'{base}_2_ref_vocals.wav'), 
                 (ref_vocals / (np.abs(ref_vocals).max() + 1e-8) * 0.9).astype(np.float32), sr)
        print(f"  [Debug] Saved intermediate vocals to {debug_dir}/")
    
    print("  Stage 3: Multi-resolution speaker separation...")
    separated = multires_speaker_separation(mic_vocals, ref_vocals, sr)
    
    if output_dir:
        sf.write(os.path.join(debug_dir, f'{base}_3_separated.wav'), 
                 (separated / (np.abs(separated).max() + 1e-8) * 0.9).astype(np.float32), sr)
    
    print("  Stage 4: Neural speech enhancement (DeepFilterNet)...")
    enhanced = run_deepfilternet(separated, sr)
    
    if output_dir:
        sf.write(os.path.join(debug_dir, f'{base}_4_enhanced.wav'), 
                 (enhanced / (np.abs(enhanced).max() + 1e-8) * 0.9).astype(np.float32), sr)
    
    print("  Stage 5: Highpass filter...")
    final = highpass_filter(enhanced, sr, cutoff=80)
    
    # Ensure same length as input
    if len(final) > len(mic):
        final = final[:len(mic)]
    elif len(final) < len(mic):
        final = np.concatenate([final, np.zeros(len(mic) - len(final))])
    
    return final


def main():
    parser = argparse.ArgumentParser(description='Advanced Demucs + Multi-resolution + DeepFilterNet processing')
    parser.add_argument('--mic', required=True, help='Path to microphone recording')
    parser.add_argument('--ref', required=True, help='Path to reference recording')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--debug', action='store_true', help='Save intermediate tracks for debugging')
    args = parser.parse_args()
    
    print(f"Advanced Processing: {args.mic}")
    print("=" * 50)
    
    # Load audio at 16kHz
    print("Loading audio...")
    mic, ref, sr, original_sr = load_audio(args.mic, args.ref, 16000)
    print(f"  Loaded: {len(mic)/sr:.1f}s @ {sr}Hz")
    
    # Normalize inputs
    mic = mic / (np.abs(mic).max() + 1e-8) * 0.95
    ref = ref / (np.abs(ref).max() + 1e-8) * 0.95
    
    # Process (pass output path for debug files if --debug flag)
    output_dir = args.output if args.debug else None
    output = advanced_process(mic, ref, sr, output_dir)
    
    # Normalize output
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.9
    
    # Save at 16kHz (good quality for speech)
    print(f"  Saving to {args.output}...")
    sf.write(args.output, output.astype(np.float32), sr)
    
    print("=" * 50)
    print("✅ Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simultaneous Speaker Separation using Direct Reference Subtraction
Since we have the EXACT reference signal, we can directly subtract it after
estimating the acoustic transfer function.
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
import argparse
import os
import torch


def load_audio(path: str, target_sr: int = 44100):
    """Load audio."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Estimate delay."""
    print("[2/6] Estimating delay...")
    chunk = min(sr * 2, len(mic))
    corr = signal.correlate(mic[:chunk], ref[:chunk], mode='same')
    delay = np.argmax(corr) - chunk // 2
    print(f"  Delay: {delay} samples ({delay*1000/sr:.1f}ms)")
    return delay


def align_signals(mic: np.ndarray, ref: np.ndarray, delay: int):
    """Align reference to mic."""
    if delay > 0:
        ref = np.concatenate([np.zeros(delay, dtype=np.float32), ref])[:len(mic)]
    elif delay < 0:
        ref = ref[-delay:]
    if len(ref) < len(mic):
        ref = np.concatenate([ref, np.zeros(len(mic) - len(ref), dtype=np.float32)])
    return ref[:len(mic)].astype(np.float32)


def run_demucs(audio: np.ndarray, orig_sr: int):
    """Run Demucs to extract vocals."""
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
    
    audio_44k = signal.resample_poly(audio, 44100, orig_sr).astype(np.float32)
    
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    vocals = signal.resample_poly(vocals, orig_sr, 44100).astype(np.float32)
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    return vocals.astype(np.float32)


def estimate_room_impulse_response(mic_vocals: np.ndarray, ref_vocals: np.ndarray, sr: int, ir_len_ms: float = 100):
    """
    Estimate the room impulse response (how singer's voice reaches your mic).
    This accounts for speaker characteristics, room acoustics, etc.
    """
    print("[5/6] Estimating acoustic transfer function...")
    
    ir_len = int(ir_len_ms * sr / 1000)
    
    # Use frequency domain for efficiency
    n_fft = 2048
    hop = 512
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_frames = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft = mic_stft[:, :min_frames]
    ref_stft = ref_stft[:, :min_frames]
    
    n_freq = mic_stft.shape[0]
    
    # Estimate transfer function H for each frequency
    # H = E[mic * conj(ref)] / E[ref * conj(ref)]
    # This is the Wiener solution
    
    eps = 1e-8
    
    # Time-varying transfer function estimation
    # Use exponential moving average
    alpha = 0.95
    
    H = np.zeros((n_freq, min_frames), dtype=np.complex64)
    num = np.zeros(n_freq, dtype=np.complex64)
    den = np.zeros(n_freq, dtype=np.float32)
    
    for t in range(min_frames):
        ref_frame = ref_stft[:, t]
        mic_frame = mic_stft[:, t]
        
        # Update running estimates
        num = alpha * num + (1 - alpha) * mic_frame * np.conj(ref_frame)
        den = alpha * den + (1 - alpha) * np.abs(ref_frame) ** 2
        
        H[:, t] = num / (den + eps)
    
    # Smooth H across time
    H_mag = np.abs(H)
    H_phase = np.angle(H)
    
    # Median filter to remove outliers
    H_mag = median_filter(H_mag, size=(3, 7))
    
    # Reconstruct H
    H = H_mag * np.exp(1j * H_phase)
    
    # Estimate singer component in mic
    singer_in_mic = H * ref_stft
    
    # Subtract to get your voice
    your_voice_stft = mic_stft - singer_in_mic
    
    # Apply soft mask to clean up artifacts
    your_mag = np.abs(your_voice_stft)
    mic_mag = np.abs(mic_stft)
    singer_mag = np.abs(singer_in_mic)
    
    # Ideal ratio mask
    mask = your_mag / (your_mag + singer_mag + eps)
    mask = np.clip(mask, 0.0, 1.0)
    
    # Smooth mask
    mask = median_filter(mask, size=(3, 5))
    
    # Apply mask to original phase
    your_phase = np.angle(your_voice_stft)
    clean_stft = your_mag * mask * np.exp(1j * your_phase)
    
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(mic_vocals)].astype(np.float32)


def wiener_postfilter(audio: np.ndarray, sr: int):
    """Final cleanup."""
    print("[6/6] Final cleanup...")
    
    n_fft = 2048
    hop = 512
    
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 15)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
    
    # Wiener gain
    gain = np.maximum(mag**2 - 2 * noise**2, 0) / (mag**2 + eps)
    gain = np.sqrt(np.clip(gain, 0.1, 1.0))
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


eps = 1e-8

def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 100):
    """Remove low frequency."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the direct subtraction pipeline."""
    print("=" * 60)
    print("Direct Reference Subtraction (Simultaneous Speech)")
    print("=" * 60)
    
    # Load
    print("[1/6] Loading audio...")
    mic, sr = load_audio(mic_path, 44100)
    ref, _ = load_audio(ref_path, 44100)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s at {sr}Hz")
    
    # Align
    delay = estimate_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Extract vocals
    print("[3/6] Extracting vocals from mic (Demucs)...")
    mic_vocals = run_demucs(mic, sr)
    
    print("[4/6] Extracting vocals from reference (Demucs)...")
    ref_vocals = run_demucs(ref_aligned, sr)
    
    # Direct subtraction with transfer function estimation
    your_voice = estimate_room_impulse_response(mic_vocals, ref_vocals, sr)
    
    # Cleanup
    cleaned = wiener_postfilter(your_voice, sr)
    final = highpass_filter(cleaned, sr, cutoff=100)
    
    # Normalize
    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.9
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, final, sr)
    print(f"\n✓ Saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mic, args.ref, args.output)



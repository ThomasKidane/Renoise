#!/usr/bin/env python3
"""
Speaker Separation using reference-guided spectral masking
Uses Demucs for vocal extraction, then separates speakers using the reference
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
import argparse
import os
import torch


def load_audio(path: str, target_sr: int = 44100):
    """Load audio at target sample rate."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = signal.resample_poly(audio, target_sr, sr).astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio, target_sr


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int):
    """Estimate delay using cross-correlation."""
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


def run_demucs(audio: np.ndarray, sr: int):
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
    
    if sr != 44100:
        audio_44k = signal.resample_poly(audio, 44100, sr).astype(np.float32)
    else:
        audio_44k = audio
    
    audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    vocals = sources[0, 3, :, :].mean(dim=0).cpu().numpy()
    
    if sr != 44100:
        vocals = signal.resample_poly(vocals, sr, 44100).astype(np.float32)
    
    vocals = vocals[:len(audio)]
    if len(vocals) < len(audio):
        vocals = np.concatenate([vocals, np.zeros(len(audio) - len(vocals))])
    
    return vocals.astype(np.float32)


def compute_voice_activity(audio: np.ndarray, sr: int, frame_ms: float = 20):
    """Compute voice activity detection mask."""
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    
    n_frames = (len(audio) - frame_len) // hop + 1
    energy = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        energy[i] = np.sqrt(np.mean(frame ** 2))
    
    # Normalize
    energy = energy / (np.max(energy) + 1e-8)
    
    # Threshold
    threshold = np.percentile(energy, 30)
    vad = energy > threshold
    
    return vad, hop


def speaker_separation_ica(mic_vocals: np.ndarray, ref_vocals: np.ndarray, sr: int):
    """
    Separate speakers using ICA-like approach in STFT domain.
    Your voice = mic_vocals - scaled(ref_vocals)
    """
    print("[5/6] ICA-based speaker separation...")
    
    n_fft = 2048
    hop = 512
    
    _, _, mic_stft = signal.stft(mic_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, ref_stft = signal.stft(ref_vocals, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    min_frames = min(mic_stft.shape[1], ref_stft.shape[1])
    mic_stft = mic_stft[:, :min_frames]
    ref_stft = ref_stft[:, :min_frames]
    
    mic_mag = np.abs(mic_stft)
    ref_mag = np.abs(ref_stft)
    mic_phase = np.angle(mic_stft)
    
    # Estimate optimal scaling factor per frequency bin
    # Find alpha that minimizes |mic - alpha * ref|^2 where ref is strong
    eps = 1e-8
    
    # Time-varying gain estimation
    # Where ref is active, estimate how much of it leaks into mic
    ref_active = ref_mag > np.percentile(ref_mag, 30)
    
    # Estimate transfer function (how much ref appears in mic)
    H = np.zeros(mic_stft.shape[0], dtype=np.complex64)
    for f in range(mic_stft.shape[0]):
        ref_f = ref_stft[f, ref_active[f, :]]
        mic_f = mic_stft[f, ref_active[f, :]]
        if len(ref_f) > 10:
            # Least squares: H = (ref^H * mic) / (ref^H * ref)
            H[f] = np.sum(np.conj(ref_f) * mic_f) / (np.sum(np.abs(ref_f)**2) + eps)
    
    # Smooth H across frequency
    H_mag = np.abs(H)
    H_mag = np.convolve(H_mag, np.ones(5)/5, mode='same')
    H_phase = np.angle(H)
    H = H_mag * np.exp(1j * H_phase)
    
    # Subtract estimated singer component
    singer_estimate = H[:, np.newaxis] * ref_stft
    your_voice_stft = mic_stft - singer_estimate
    
    # Soft masking to clean up
    your_mag = np.abs(your_voice_stft)
    your_phase = np.angle(your_voice_stft)
    
    # Where ref is very strong and your estimate is weak, suppress more
    mask = your_mag / (your_mag + 0.5 * ref_mag + eps)
    mask = np.clip(mask, 0.1, 1.0)
    
    # Smooth mask
    mask = median_filter(mask, size=(3, 5))
    
    clean_stft = your_mag * mask * np.exp(1j * your_phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(mic_vocals)].astype(np.float32)


def wiener_cleanup(audio: np.ndarray, sr: int):
    """Final Wiener filter cleanup."""
    print("[6/6] Final cleanup...")
    
    n_fft = 2048
    hop = 512
    
    _, _, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    frame_energy = np.sum(mag**2, axis=0)
    n_quiet = max(1, len(frame_energy) // 20)
    quiet_idx = np.argsort(frame_energy)[:n_quiet]
    noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
    
    gain = np.maximum(mag**2 - noise**2, 0) / (mag**2 + 1e-8)
    gain = np.sqrt(np.clip(gain, 0.15, 1.0))
    
    clean_stft = mag * gain * np.exp(1j * phase)
    _, clean = signal.istft(clean_stft, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    
    return clean[:len(audio)].astype(np.float32)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80):
    """Remove low frequency rumble."""
    sos = signal.butter(4, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfilt(sos, audio).astype(np.float32)


def run_pipeline(mic_path: str, ref_path: str, output_path: str):
    """Run the speaker separation pipeline."""
    print("=" * 60)
    print("Speaker Separation: Your Voice vs Singer")
    print("=" * 60)
    
    # Step 1: Load
    print("[1/6] Loading audio...")
    mic, sr = load_audio(mic_path, 44100)
    ref, _ = load_audio(ref_path, 44100)
    
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    print(f"  Loaded {len(mic)/sr:.1f}s at {sr}Hz")
    
    # Step 2: Align
    delay = estimate_delay(mic, ref, sr)
    ref_aligned = align_signals(mic, ref, delay)
    
    # Step 3: Extract vocals from mic
    print("[3/6] Extracting vocals from mic (Demucs)...")
    mic_vocals = run_demucs(mic, sr)
    
    # Step 4: Extract vocals from reference
    print("[4/6] Extracting vocals from reference (Demucs)...")
    ref_vocals = run_demucs(ref_aligned, sr)
    
    # Step 5: Separate your voice from singer
    your_voice = speaker_separation_ica(mic_vocals, ref_vocals, sr)
    
    # Step 6: Cleanup
    cleaned = wiener_cleanup(your_voice, sr)
    final = highpass_filter(cleaned, sr, cutoff=80)
    
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



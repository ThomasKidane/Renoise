# Renoise 

**Voice isolation that removes music, calls, and system audio from your microphone using deep learning and signal processing.**

Renoise uses a multi-stage pipeline combining source separation (Demucs), acoustic echo cancellation (DTRL-AEC), and neural speech enhancement (DeepFilterNet) to separate *your voice* from everything playing on your computer—even when you're speaking over music with lyrics.

## The Problem

When you use voice dictation while playing audio:

- Your mic picks up both your voice AND the system audio (music, podcasts, calls)
- Voice assistants transcribe everything, making the output unusable
- Traditional echo cancellation can't handle music with vocals

## The Solution

Renoise uses a **multi-stage pipeline**:

1. **Demucs** (Meta's music source separation) — Extracts vocals from both your mic and the reference
2. **DTRL-AEC** (from Microsoft's AEC Challenge) — Acoustic echo cancellation with deep transform learning
3. **Multi-resolution Speaker Separation** — Separates YOUR voice from the singer's voice
4. **DeepFilterNet** — Neural speech enhancement (DNS Challenge winner)

### Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│   Demucs    │────▶│  Speaker    │────▶│ DeepFilter  │────▶ Your Voice
│  (voice +   │     │  (extract   │     │ Separation  │     │    Net      │      (clean)
│   music)    │     │   vocals)   │     │ (DTRL-AEC)  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           ▲                   ▲
                           │                   │
                    ┌──────┴───────────────────┴──────┐
                    │         Reference Audio          │
                    │  (what's playing on your Mac)   │
                    └─────────────────────────────────┘
```

## Features

- 🎵 **Works with music** — Even songs with vocals
- 🔊 **Removes any system audio** — Music, podcasts, calls, videos
- 🎯 **Speaker separation** — Distinguishes your voice from singers
- 📊 **Debug tracks** — See each pipeline stage for quality analysis
- 🖥️ **macOS-native UI** — Modern Electron interface
- ⚡ **GPU accelerated** — Uses Apple Silicon MPS for fast processing

## Download

### macOS (Apple Silicon)

Download the latest `.dmg` from [Releases](https://github.com/ThomasKidane/Renoise/releases).

### Build from Source

```bash
# Clone
git clone https://github.com/ThomasKidane/Renoise.git
cd Renoise

# Install Node dependencies
npm install

# Install Python dependencies
pip install demucs deepfilternet soundfile numpy scipy torch torchaudio

# Run in development
npm run dev

# Build DMG for distribution
npm run dist
```

## Audio Routing Setup (macOS)

Renoise needs to capture your system audio. Here's how:

### 1. Install BlackHole

```bash
brew install blackhole-2ch blackhole-16ch
```

### 2. Create Multi-Output Device

1. Open **Audio MIDI Setup** (search in Spotlight)
2. Click **+** → **Create Multi-Output Device**
3. Check both:
   - Your speakers/headphones
   - BlackHole 2ch
4. Set this as your system output

### 3. Configure Renoise

- **Input**: MacBook Pro Microphone (your voice)
- **Reference**: BlackHole 2ch (captures system audio)
- **Output**: BlackHole 16ch (optional, for routing to other apps)

## How It Works

### Recording Mode

1. Click **Record** to start capturing
2. Play music, speak over it
3. Stop recording
4. Renoise processes your audio through:
   - Demucs (extracts vocals from mic)
   - Demucs (extracts vocals from reference — the singer)
   - DTRL-AEC + multi-resolution separation (subtracts singer from your vocals)
   - DeepFilterNet (neural speech enhancement)
5. Listen to the result in the **Recordings** tab

### Debug Tracks

Each recording shows intermediate pipeline stages:

- **Mic Vocals** — Your voice + singer (after Demucs)
- **Reference Vocals** — Just the singer
- **After Separation** — Your voice only
- **After DeepFilterNet** — Final enhanced output

## Tech Stack

- **Electron + TypeScript/React** — Cross-platform desktop app
- **Demucs** (Meta) — Music source separation
- **DTRL-AEC** — Deep transform learning for acoustic echo cancellation (Microsoft AEC Challenge)
- **DeepFilterNet** — Neural speech enhancement (DNS Challenge winner)
- **naudiodon2** — Low-latency native audio I/O
- **Python + PyTorch** — ML inference pipeline
- **C++** — Native audio processing

## Why I Built This

I use [Wispr](https://wispr.ai) for voice dictation and dedicated a mouse button to it (keyboard defeats the purpose of voice input!). But when I listen to music while working, Wispr transcribes the lyrics along with my voice.

Renoise solves this by isolating just my voice, even when I'm speaking over music with vocals.

## Roadmap

- [ ] Real-time processing (currently recording-only)
- [ ] Windows/Linux support
- [ ] Voice activity detection
- [ ] Automatic gain control
- [ ] Integration with Wispr/other voice apps

## License

MIT

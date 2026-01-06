# Renoise 🎤

**AI-powered voice isolation that removes music, calls, and system audio from your microphone.**

Renoise uses state-of-the-art deep learning to separate *your voice* from everything playing on your computer—even when you're speaking over music with lyrics.

![Renoise Screenshot](resources/screenshot.png)

## The Problem

When you use voice dictation while playing audio:
- Your mic picks up both your voice AND the system audio (music, podcasts, calls)
- Voice assistants transcribe everything, making the output unusable
- Traditional echo cancellation can't handle music with vocals

## The Solution

Renoise uses a **multi-stage deep learning pipeline**:

1. **Demucs** (Meta's music source separation) - Extracts vocals from both your mic and the reference
2. **Multi-resolution Speaker Separation** - Separates YOUR voice from the singer's voice
3. **DeepFilterNet** - State-of-the-art neural speech enhancement

### Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│   Demucs    │────▶│  Speaker    │────▶│ DeepFilter  │────▶ Your Voice
│  (voice +   │     │  (extract   │     │ Separation  │     │    Net      │      (clean)
│   music)    │     │   vocals)   │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           ▲                   ▲
                           │                   │
                    ┌──────┴───────────────────┴──────┐
                    │         Reference Audio          │
                    │  (what's playing on your Mac)   │
                    └─────────────────────────────────┘
```

## Features

- 🎵 **Works with music** - Even songs with vocals
- 🔊 **Removes any system audio** - Music, podcasts, calls, videos
- 🎯 **Speaker separation** - Distinguishes your voice from singers
- 📊 **Debug tracks** - See each pipeline stage for quality analysis
- 🖥️ **Beautiful UI** - Modern macOS-native interface
- ⚡ **GPU accelerated** - Uses Apple Silicon MPS for fast processing

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

# Install Python dependencies (for Demucs/DeepFilterNet)
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
   - Demucs (extracts vocals from reference - the singer)
   - Multi-resolution separation (subtracts singer from your vocals)
   - DeepFilterNet (neural speech enhancement)
5. Listen to the result in the **Recordings** tab

### Debug Tracks
Each recording shows intermediate pipeline stages:
- **Mic Vocals** - Your voice + singer (after Demucs)
- **Reference Vocals** - Just the singer
- **After Separation** - Your voice only
- **After DeepFilterNet** - Final enhanced output

## Tech Stack

- **Electron** - Cross-platform desktop app
- **Demucs** (Meta) - State-of-the-art music source separation
- **DeepFilterNet** - Neural speech enhancement (DNS Challenge winner)
- **naudiodon2** - Low-latency audio I/O
- **TypeScript/React** - Modern UI

## Why I Built This

I use [Wispr](https://wispr.ai) for voice dictation and dedicated a mouse button to it (keyboard defeats the purpose of voice input!). But when I listen to music while working, Wispr transcribes the lyrics along with my voice.

Renoise solves this by using deep learning to isolate just my voice, even when I'm speaking over music with vocals.

## Roadmap

- [ ] Real-time processing (currently recording-only)
- [ ] Windows/Linux support
- [ ] Voice activity detection
- [ ] Automatic gain control
- [ ] Integration with Wispr/other voice apps

## License

MIT

---

*Built with ❤️ for voice-first computing*

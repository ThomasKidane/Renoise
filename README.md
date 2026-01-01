# Renoise 🎤

**Real-time voice isolation for voice-first computing.**

Renoise removes system audio (music, calls, podcasts) from your microphone input in real-time, so voice assistants like [Wispr](https://wispr.ai) only hear *your voice*.

## The Problem

When you use voice dictation with speakers playing audio:
- Your mic picks up both your voice AND the system audio
- Voice assistants transcribe the music/podcast along with your speech
- The transcription becomes unusable

## The Solution

Renoise uses a **cascaded deep learning pipeline**:

1. **DTLN-AEC** (Deep Learning Acoustic Echo Cancellation) - Removes the bulk of the reference audio using a neural network trained on the Microsoft AEC Challenge
2. **Spectral Cleanup** - Removes any residual artifacts through adaptive spectral subtraction

### How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│   DTLN-AEC  │────▶│  Spectral   │────▶ Clean Voice
│   (voice +  │     │  (removes   │     │  Cleanup    │
│   echo)     │     │   echo)     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                           ▲
                           │
                    ┌──────┴──────┐
                    │  Reference  │
                    │  (system    │
                    │   audio)    │
                    └─────────────┘
```

## Features

- **Real-time processing** - DTLN-AEC runs live on your audio
- **High-quality recordings** - Cascaded pipeline for best quality
- **Works with any audio** - Music, podcasts, calls, videos
- **macOS native** - Uses BlackHole for system audio capture

## Requirements

- macOS
- [BlackHole](https://existential.audio/blackhole/) (2ch for reference, 16ch for output)
- Node.js 18+
- Python 3.11+ with TensorFlow

## Setup

```bash
# Clone
git clone https://github.com/ThomasKidane/Renoise.git
cd Renoise

# Install dependencies
npm install

# Set up DTLN-AEC
cd DTRL-AEC/DTLN-aec
python3 -m venv venv
source venv/bin/activate
pip install tensorflow soundfile numpy scipy

# Run
cd ../..
npm run dev
```

## Audio Routing (macOS)

1. Install BlackHole 2ch and 16ch
2. Create a Multi-Output Device (System Preferences → Audio MIDI Setup):
   - Add your speakers + BlackHole 2ch
3. Set system output to the Multi-Output Device
4. In Renoise:
   - Input: Your microphone
   - Reference: BlackHole 2ch (captures system audio)
   - Output: BlackHole 16ch (clean voice output)

## Tech Stack

- **Electron** - Desktop app
- **DTLN-AEC** - Deep learning echo cancellation (TensorFlow Lite)
- **naudiodon2** - Low-latency audio I/O
- **TypeScript/React** - UI

## Why I Built This

I use [Wispr](https://wispr.ai) for voice dictation, but I also listen to music while working. The problem: Wispr would transcribe my music along with my voice.

I dedicated a mouse button to Wispr (keyboard defeats the purpose of voice input!) and built Renoise to solve the echo problem.

## Future Ideas

- [ ] Noise gate based on voice activity detection
- [ ] Automatic gain control
- [ ] Support for more audio backends
- [ ] Windows/Linux support

## License

MIT

---

*Built with ❤️ for voice-first computing*

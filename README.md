# Wispr Audio Separator

A macOS menubar application that creates a virtual microphone providing clean, voice-only audio by removing system audio (music, videos, etc.) from your microphone input using adaptive filtering.

![Wispr Audio Separator](https://img.shields.io/badge/platform-macOS-blue) ![Electron](https://img.shields.io/badge/electron-28.x-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 The Problem

Voice-to-text tools like Wispr Flow struggle when there's music or audio playing from your computer. The audio bleeds into your microphone and confuses the transcription.

## 💡 The Solution

This app intercepts your microphone signal, uses the system audio as a reference, and applies an **NLMS (Normalized Least Mean Squares) adaptive filter** to subtract the known audio from your mic input. The result is a clean voice signal sent to a virtual microphone that Wispr Flow (or any other app) can use.

```
Real Microphone ──────────────────┐
                                  ▼
                    ┌──────────────────────────┐
System Audio ──────►│   Adaptive Filter (NLMS)  │──► Virtual Microphone ──► Wispr Flow
(via BlackHole)    └──────────────────────────┘
```

## 🔧 Prerequisites

### 1. Install BlackHole (Virtual Audio Driver)

```bash
brew install blackhole-2ch
```

For best results, also install the 16-channel version:
```bash
brew install blackhole-16ch
```

### 2. Configure Audio MIDI Setup

1. Open **Audio MIDI Setup** (search in Spotlight)
2. Click the **+** button at the bottom left
3. Select **Create Multi-Output Device**
4. Check both your speakers AND **BlackHole 2ch**
5. Set this Multi-Output Device as your system output in System Preferences → Sound

This routes all system audio to both your speakers AND BlackHole for capture.

### 3. Audio Routing Setup

| Signal | Device |
|--------|--------|
| **Microphone Input** | Your real microphone (Built-in, USB, etc.) |
| **Reference Input** | BlackHole 2ch (receives system audio copy) |
| **Output** | BlackHole 16ch (clean voice output) |
| **Wispr Flow Input** | BlackHole 16ch |

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wispr-audio-separator.git
cd wispr-audio-separator

# Install dependencies
npm install

# Start development mode
npm run dev

# Or build for production
npm run build
```

## 📖 Usage

1. **Launch the app** - It will appear in your menubar
2. **Click the tray icon** to open the configuration window
3. **Select your devices**:
   - **Microphone**: Your real microphone
   - **Reference**: BlackHole 2ch (system audio loopback)
   - **Output**: BlackHole 16ch (or another virtual device)
4. **Click the power button** to start processing
5. **Configure Wispr Flow** to use "BlackHole 16ch" as its microphone input

### Parameter Tuning

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Input Gain** | 0-2x | Adjust microphone sensitivity |
| **Reference Gain** | 0-2x | Adjust system audio reference level |
| **Step Size (μ)** | 0.01-1.0 | Filter learning rate. Higher = faster adaptation but less stable |
| **Delay** | 0-100ms | Compensate for speaker-to-mic propagation delay |

### Tips for Best Results

- **Start with default settings** and adjust if needed
- **Increase Step Size** if the filter isn't adapting fast enough to music changes
- **Decrease Step Size** if you hear artifacts or the filter is unstable
- **Adjust Delay** if you notice the filter isn't canceling audio well (try 10-30ms)
- **Reset Filter** if it gets stuck or produces strange output

## 🏗️ Architecture

### Core Technologies

- **Electron** - Cross-platform desktop framework
- **naudiodon2** - Node.js bindings for PortAudio (low-level audio I/O)
- **React** - UI framework
- **TypeScript** - Type safety

### NLMS Adaptive Filter

The heart of the system is the Normalized Least Mean Squares algorithm:

```typescript
// For each sample:
y = weights · referenceBuffer  // Estimate of leaked audio
e = micInput - y               // Error = voice estimate
weights += (μ * e / ||ref||²) * referenceBuffer  // Update weights
output = e                     // Clean voice signal
```

The filter continuously adapts to estimate how system audio appears in the microphone (after going through speakers and room acoustics) and subtracts it.

### File Structure

```
src/
├── main/
│   ├── index.ts              # Electron main process
│   ├── audio-processor.ts    # Audio capture and processing
│   ├── nlms-filter.ts        # NLMS implementation
│   └── device-manager.ts     # Audio device enumeration
├── renderer/
│   ├── App.tsx               # React UI
│   ├── styles.css            # Styling
│   └── components/
│       ├── DeviceSelector.tsx
│       ├── LevelMeter.tsx
│       ├── ParameterSlider.tsx
│       └── StatusIndicator.tsx
└── preload/
    └── index.ts              # Electron preload script
```

## 🐛 Troubleshooting

### "No audio devices found"
- Make sure BlackHole is installed correctly
- Try restarting the app or your computer

### "Filter not canceling audio"
- Ensure your Multi-Output Device is set as system output
- Check that BlackHole 2ch is selected as the reference device
- Try adjusting the delay parameter (10-30ms is typical)
- Make sure system audio is actually playing

### "Voice sounds distorted"
- Reduce the Step Size parameter
- Check input/reference gain levels
- Click "Reset Filter" to start fresh

### "High CPU usage"
- This is normal during active processing
- The app uses real-time audio processing which requires CPU

## 📝 Development

```bash
# Run in development mode with hot reload
npm run dev

# Build for production
npm run build

# Package for distribution
npm run package
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [BlackHole](https://github.com/ExistentialAudio/BlackHole) - Virtual audio driver
- [naudiodon2](https://www.npmjs.com/package/naudiodon2) - Node.js audio I/O
- [Electron](https://www.electronjs.org/) - Desktop app framework

## 📚 References

- [NLMS Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Least_mean_squares_filter#Normalised_least_mean_squares_filter_(NLMS))
- [Acoustic Echo Cancellation](https://en.wikipedia.org/wiki/Echo_cancellation)




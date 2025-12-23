/**
 * Real RNNoise Integration for Browser
 * Uses @jitsi/rnnoise-wasm - battle-tested in Jitsi Meet
 */

// RNNoise constants
const RNNOISE_SAMPLE_RATE = 48000;
const RNNOISE_FRAME_SIZE = 480; // 10ms at 48kHz

// AEC3 Server URL
const AEC3_SERVER = 'http://localhost:8081';

/**
 * WebRTC AEC3 Processor - Uses the real AEC3 via server
 */
class AEC3Processor {
  constructor() {
    this.serverAvailable = false;
  }

  async checkServer() {
    try {
      const response = await fetch(`${AEC3_SERVER}/health`);
      const data = await response.json();
      this.serverAvailable = data.aec3Available;
      return this.serverAvailable;
    } catch (e) {
      console.error('AEC3 server not available:', e.message);
      this.serverAvailable = false;
      return false;
    }
  }

  async process(input, reference) {
    // Check server availability
    if (!this.serverAvailable) {
      await this.checkServer();
    }
    
    if (!this.serverAvailable) {
      console.warn('AEC3 server not available, returning input unchanged');
      console.warn('Start the server with: node test-app/aec3-server.js');
      return input;
    }

    try {
      // Convert Float32Arrays to WAV format
      const inputWav = this.createWavBuffer(input, 48000);
      const refWav = this.createWavBuffer(reference, 48000);
      
      // Send to server
      const response = await fetch(`${AEC3_SERVER}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: this.arrayBufferToBase64(inputWav),
          reference: this.arrayBufferToBase64(refWav)
        })
      });
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'AEC3 processing failed');
      }
      
      // Decode output WAV
      const outputWav = this.base64ToArrayBuffer(result.output);
      const output = this.parseWavBuffer(outputWav);
      
      return output;
      
    } catch (error) {
      console.error('AEC3 processing error:', error);
      return input;
    }
  }

  createWavBuffer(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    
    // WAV header
    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    this.writeString(view, 8, 'WAVE');
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this.writeString(view, 36, 'data');
    view.setUint32(40, numSamples * 2, true);
    
    // Convert float to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
    
    return buffer;
  }

  parseWavBuffer(buffer) {
    const view = new DataView(buffer);
    
    // Skip header, find data chunk
    let offset = 12;
    while (offset < buffer.byteLength - 8) {
      const chunkId = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
      );
      const chunkSize = view.getUint32(offset + 4, true);
      
      if (chunkId === 'data') {
        offset += 8;
        const numSamples = chunkSize / 2;
        const samples = new Float32Array(numSamples);
        
        for (let i = 0; i < numSamples; i++) {
          const sample = view.getInt16(offset + i * 2, true);
          samples[i] = sample / 32768;
        }
        
        return samples;
      }
      
      offset += 8 + chunkSize;
    }
    
    return new Float32Array(0);
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }
}

// Global module reference
let rnnoiseModule = null;
let rnnoiseLoading = null;

/**
 * Load the RNNoise WASM module
 */
async function loadRNNoise() {
  if (rnnoiseModule) return rnnoiseModule;
  if (rnnoiseLoading) return rnnoiseLoading;
  
  rnnoiseLoading = new Promise(async (resolve, reject) => {
    try {
      // Load the WASM module
      const script = document.createElement('script');
      script.src = 'rnnoise-wasm.js';
      
      script.onload = async () => {
        try {
          // createRNNWasmModule is exposed globally by the script
          const module = await window.createRNNWasmModule();
          rnnoiseModule = module;
          console.log('RNNoise WASM module loaded successfully');
          resolve(module);
        } catch (e) {
          console.error('Failed to initialize RNNoise module:', e);
          reject(e);
        }
      };
      
      script.onerror = (e) => {
        console.error('Failed to load RNNoise script:', e);
        reject(new Error('Failed to load RNNoise WASM script'));
      };
      
      document.head.appendChild(script);
    } catch (e) {
      reject(e);
    }
  });
  
  return rnnoiseLoading;
}

/**
 * Real RNNoise Processor using WASM
 */
class RealRNNoiseProcessor {
  constructor() {
    this.module = null;
    this.state = null;
    this.inputBuffer = null;
    this.outputBuffer = null;
    this.loaded = false;
  }

  async initialize() {
    try {
      this.module = await loadRNNoise();
      
      // Create RNNoise state
      this.state = this.module._rnnoise_create(null);
      
      // Allocate buffers in WASM memory
      // RNNoise expects float32 samples, FRAME_SIZE floats = FRAME_SIZE * 4 bytes
      this.inputBuffer = this.module._malloc(RNNOISE_FRAME_SIZE * 4);
      this.outputBuffer = this.module._malloc(RNNOISE_FRAME_SIZE * 4);
      
      this.loaded = true;
      console.log('RNNoise processor initialized');
      return true;
    } catch (e) {
      console.error('Failed to initialize RNNoise:', e);
      this.loaded = false;
      return false;
    }
  }

  /**
   * Process a single frame of audio
   * @param {Float32Array} frame - 480 samples at 48kHz
   * @returns {Float32Array} - Denoised frame
   */
  processFrame(frame) {
    if (!this.loaded || !this.state) {
      return frame;
    }

    // Copy input to WASM memory
    // RNNoise expects samples in range [-32768, 32767] (int16 scale)
    const inputView = new Float32Array(this.module.HEAPF32.buffer, this.inputBuffer, RNNOISE_FRAME_SIZE);
    for (let i = 0; i < RNNOISE_FRAME_SIZE; i++) {
      inputView[i] = (frame[i] || 0) * 32768;
    }

    // Process with RNNoise
    // Returns VAD probability (0-1)
    const vadProb = this.module._rnnoise_process_frame(this.state, this.outputBuffer, this.inputBuffer);

    // Copy output from WASM memory and scale back
    const outputView = new Float32Array(this.module.HEAPF32.buffer, this.outputBuffer, RNNOISE_FRAME_SIZE);
    const output = new Float32Array(RNNOISE_FRAME_SIZE);
    for (let i = 0; i < RNNOISE_FRAME_SIZE; i++) {
      output[i] = outputView[i] / 32768;
    }

    return output;
  }

  /**
   * Process full audio buffer
   * For echo cancellation: we process the difference between input and reference
   */
  process(input, reference) {
    if (!this.loaded) {
      console.warn('RNNoise not loaded, returning input unchanged');
      return input;
    }

    const output = new Float32Array(input.length);
    const numFrames = Math.floor(input.length / RNNOISE_FRAME_SIZE);

    for (let i = 0; i < numFrames; i++) {
      const start = i * RNNOISE_FRAME_SIZE;
      
      // Create frame with reference subtraction
      const frame = new Float32Array(RNNOISE_FRAME_SIZE);
      for (let j = 0; j < RNNOISE_FRAME_SIZE; j++) {
        const idx = start + j;
        // Subtract reference (echo) and let RNNoise clean up residual
        frame[j] = (input[idx] || 0) - 0.8 * (reference[idx] || 0);
      }
      
      // Process through RNNoise
      const processed = this.processFrame(frame);
      
      // Copy to output
      for (let j = 0; j < RNNOISE_FRAME_SIZE; j++) {
        output[start + j] = processed[j];
      }
    }

    // Handle remaining samples (pad with zeros)
    const remaining = input.length % RNNOISE_FRAME_SIZE;
    if (remaining > 0) {
      const start = numFrames * RNNOISE_FRAME_SIZE;
      const frame = new Float32Array(RNNOISE_FRAME_SIZE);
      for (let j = 0; j < remaining; j++) {
        frame[j] = (input[start + j] || 0) - 0.8 * (reference[start + j] || 0);
      }
      const processed = this.processFrame(frame);
      for (let j = 0; j < remaining; j++) {
        output[start + j] = processed[j];
      }
    }

    return output;
  }

  destroy() {
    if (this.module && this.state) {
      this.module._rnnoise_destroy(this.state);
      this.state = null;
    }
    if (this.module && this.inputBuffer) {
      this.module._free(this.inputBuffer);
      this.inputBuffer = null;
    }
    if (this.module && this.outputBuffer) {
      this.module._free(this.outputBuffer);
      this.outputBuffer = null;
    }
    this.loaded = false;
  }
}

/**
 * RNNoise + NLMS Hybrid
 * Uses NLMS for echo cancellation, then RNNoise for noise suppression
 */
class RNNoiseHybridProcessor {
  constructor(filterLength = 512, stepSize = 0.3) {
    this.filterLength = filterLength;
    this.mu = stepSize;
    this.weights = new Float32Array(filterLength);
    this.buffer = new Float32Array(filterLength);
    this.bufferIdx = 0;
    
    this.rnnoise = new RealRNNoiseProcessor();
    this.rnnoiseReady = false;
  }

  async initialize() {
    this.rnnoiseReady = await this.rnnoise.initialize();
    return this.rnnoiseReady;
  }

  process(input, reference) {
    // First pass: NLMS adaptive filter for echo cancellation
    const nlmsOutput = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.bufferIdx] = reference[i];
      
      // Filter output (echo estimate)
      let echoEst = 0;
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIdx - j + this.filterLength) % this.filterLength;
        echoEst += this.weights[j] * this.buffer[idx];
      }
      
      const error = input[i] - echoEst;
      nlmsOutput[i] = error;
      
      // Update weights
      let norm = 1e-8;
      for (let j = 0; j < this.filterLength; j++) {
        norm += this.buffer[j] ** 2;
      }
      
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIdx - j + this.filterLength) % this.filterLength;
        this.weights[j] += this.mu / norm * error * this.buffer[idx];
      }
      
      this.bufferIdx = (this.bufferIdx + 1) % this.filterLength;
    }

    // Second pass: RNNoise for residual noise suppression
    if (this.rnnoiseReady) {
      return this.processWithRNNoise(nlmsOutput);
    }
    
    return nlmsOutput;
  }

  processWithRNNoise(input) {
    const output = new Float32Array(input.length);
    const numFrames = Math.floor(input.length / RNNOISE_FRAME_SIZE);

    for (let i = 0; i < numFrames; i++) {
      const start = i * RNNOISE_FRAME_SIZE;
      const frame = input.slice(start, start + RNNOISE_FRAME_SIZE);
      const processed = this.rnnoise.processFrame(frame);
      output.set(processed, start);
    }

    // Handle remaining
    const remaining = input.length % RNNOISE_FRAME_SIZE;
    if (remaining > 0) {
      const start = numFrames * RNNOISE_FRAME_SIZE;
      const frame = new Float32Array(RNNOISE_FRAME_SIZE);
      frame.set(input.slice(start, start + remaining));
      const processed = this.rnnoise.processFrame(frame);
      for (let j = 0; j < remaining; j++) {
        output[start + j] = processed[j];
      }
    }

    return output;
  }

  destroy() {
    this.rnnoise.destroy();
  }
}

/**
 * Spectral Gate with Voice Activity Detection
 */
class SpectralGateVAD {
  constructor(fftSize = 2048, voiceMinFreq = 80, voiceMaxFreq = 3500) {
    this.fftSize = fftSize;
    this.hopSize = fftSize / 4;
    this.voiceMinFreq = voiceMinFreq;
    this.voiceMaxFreq = voiceMaxFreq;
    this.noiseFloor = null;
    this.smoothing = 0.95;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const fftSize = this.fftSize;
    const hopSize = this.hopSize;
    
    const window = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));
    }
    
    const numFrames = Math.floor((input.length - fftSize) / hopSize) + 1;
    
    if (!this.noiseFloor) {
      this.noiseFloor = new Float32Array(fftSize / 2 + 1).fill(0.001);
    }
    
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopSize;
      
      const inputFrame = new Float32Array(fftSize);
      const refFrame = new Float32Array(fftSize);
      
      for (let i = 0; i < fftSize && start + i < input.length; i++) {
        inputFrame[i] = input[start + i] * window[i];
        refFrame[i] = reference[start + i] * window[i];
      }
      
      const inputSpec = this.dft(inputFrame);
      const refSpec = this.dft(refFrame);
      
      const inputMag = new Float32Array(fftSize / 2 + 1);
      const refMag = new Float32Array(fftSize / 2 + 1);
      
      for (let k = 0; k <= fftSize / 2; k++) {
        inputMag[k] = Math.sqrt(inputSpec.re[k] ** 2 + inputSpec.im[k] ** 2);
        refMag[k] = Math.sqrt(refSpec.re[k] ** 2 + refSpec.im[k] ** 2);
      }
      
      for (let k = 0; k <= fftSize / 2; k++) {
        this.noiseFloor[k] = this.smoothing * this.noiseFloor[k] + (1 - this.smoothing) * refMag[k];
      }
      
      const sampleRate = 48000;
      const outputSpec = { re: new Float32Array(fftSize), im: new Float32Array(fftSize) };
      
      for (let k = 0; k <= fftSize / 2; k++) {
        const freq = k * sampleRate / fftSize;
        const isVoiceBand = freq >= this.voiceMinFreq && freq <= this.voiceMaxFreq;
        
        const snr = inputMag[k] / (this.noiseFloor[k] + 1e-10);
        let gain = Math.max(0, 1 - 1.5 / snr);
        
        if (isVoiceBand) {
          gain = Math.max(gain, 0.2);
        } else {
          gain *= 0.3;
        }
        
        outputSpec.re[k] = inputSpec.re[k] * gain;
        outputSpec.im[k] = inputSpec.im[k] * gain;
        
        if (k > 0 && k < fftSize / 2) {
          outputSpec.re[fftSize - k] = outputSpec.re[k];
          outputSpec.im[fftSize - k] = -outputSpec.im[k];
        }
      }
      
      const outputFrame = this.idft(outputSpec);
      
      for (let i = 0; i < fftSize && start + i < output.length; i++) {
        output[start + i] += outputFrame[i] * window[i];
      }
    }
    
    return output;
  }

  dft(x) {
    const N = x.length;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        re[k] += x[n] * Math.cos(angle);
        im[k] += x[n] * Math.sin(angle);
      }
    }
    return { re, im };
  }

  idft(spec) {
    const N = spec.re.length;
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < N; k++) {
        const angle = 2 * Math.PI * k * n / N;
        x[n] += spec.re[k] * Math.cos(angle) - spec.im[k] * Math.sin(angle);
      }
      x[n] /= N;
    }
    return x;
  }
}

/**
 * HPSS - Harmonic-Percussive Source Separation
 */
class HPSSProcessor {
  constructor(fftSize = 2048, harmonicKernel = 17, percussiveKernel = 17) {
    this.fftSize = fftSize;
    this.hopSize = fftSize / 4;
    this.harmonicKernel = harmonicKernel;
    this.percussiveKernel = percussiveKernel;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const fftSize = this.fftSize;
    const hopSize = this.hopSize;
    
    const window = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));
    }
    
    const numFrames = Math.floor((input.length - fftSize) / hopSize) + 1;
    
    // Build spectrogram
    const spectrogram = [];
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopSize;
      const inputFrame = new Float32Array(fftSize);
      
      for (let i = 0; i < fftSize && start + i < input.length; i++) {
        inputFrame[i] = input[start + i] * window[i];
      }
      
      const spec = this.dft(inputFrame);
      const mag = new Float32Array(fftSize / 2 + 1);
      const phase = new Float32Array(fftSize / 2 + 1);
      
      for (let k = 0; k <= fftSize / 2; k++) {
        mag[k] = Math.sqrt(spec.re[k] ** 2 + spec.im[k] ** 2);
        phase[k] = Math.atan2(spec.im[k], spec.re[k]);
      }
      
      spectrogram.push({ mag, phase });
    }
    
    // Median filtering
    const harmonicSpec = [];
    const percussiveSpec = [];
    
    for (let frame = 0; frame < numFrames; frame++) {
      const harmonicMag = new Float32Array(fftSize / 2 + 1);
      const percussiveMag = new Float32Array(fftSize / 2 + 1);
      
      for (let k = 0; k <= fftSize / 2; k++) {
        const timeWindow = [];
        const halfKernel = Math.floor(this.harmonicKernel / 2);
        for (let t = -halfKernel; t <= halfKernel; t++) {
          const frameIdx = Math.max(0, Math.min(numFrames - 1, frame + t));
          timeWindow.push(spectrogram[frameIdx].mag[k]);
        }
        timeWindow.sort((a, b) => a - b);
        harmonicMag[k] = timeWindow[Math.floor(timeWindow.length / 2)];
        
        const freqWindow = [];
        const halfKernelF = Math.floor(this.percussiveKernel / 2);
        for (let f = -halfKernelF; f <= halfKernelF; f++) {
          const freqIdx = Math.max(0, Math.min(fftSize / 2, k + f));
          freqWindow.push(spectrogram[frame].mag[freqIdx]);
        }
        freqWindow.sort((a, b) => a - b);
        percussiveMag[k] = freqWindow[Math.floor(freqWindow.length / 2)];
      }
      
      harmonicSpec.push(harmonicMag);
      percussiveSpec.push(percussiveMag);
    }
    
    // Reconstruct with harmonic mask
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopSize;
      const outputSpec = { re: new Float32Array(fftSize), im: new Float32Array(fftSize) };
      
      for (let k = 0; k <= fftSize / 2; k++) {
        const h = harmonicSpec[frame][k];
        const p = percussiveSpec[frame][k];
        
        const harmonicMask = h / (h + p + 1e-10);
        const gain = Math.pow(harmonicMask, 2);
        
        const mag = spectrogram[frame].mag[k] * gain;
        const phase = spectrogram[frame].phase[k];
        
        outputSpec.re[k] = mag * Math.cos(phase);
        outputSpec.im[k] = mag * Math.sin(phase);
        
        if (k > 0 && k < fftSize / 2) {
          outputSpec.re[fftSize - k] = outputSpec.re[k];
          outputSpec.im[fftSize - k] = -outputSpec.im[k];
        }
      }
      
      const outputFrame = this.idft(outputSpec);
      
      for (let i = 0; i < fftSize && start + i < output.length; i++) {
        output[start + i] += outputFrame[i] * window[i];
      }
    }
    
    return output;
  }

  dft(x) {
    const N = x.length;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        re[k] += x[n] * Math.cos(angle);
        im[k] += x[n] * Math.sin(angle);
      }
    }
    return { re, im };
  }

  idft(spec) {
    const N = spec.re.length;
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < N; k++) {
        const angle = 2 * Math.PI * k * n / N;
        x[n] += spec.re[k] * Math.cos(angle) - spec.im[k] * Math.sin(angle);
      }
      x[n] /= N;
    }
    return x;
  }
}

// Export algorithms
window.deepLearningAlgorithms = {
  aec3: {
    name: 'WebRTC AEC3 (Real)',
    create: () => new AEC3Processor(),
    defaultParams: {},
    async: true,
    serverBased: true
  },
  rnnoise: {
    name: 'RNNoise (Real WASM)',
    create: async () => {
      const proc = new RealRNNoiseProcessor();
      await proc.initialize();
      return proc;
    },
    defaultParams: {},
    async: true
  },
  rnnoiseHybrid: {
    name: 'RNNoise + NLMS Hybrid',
    create: async (params) => {
      const proc = new RNNoiseHybridProcessor(params.filterLength, params.stepSize);
      await proc.initialize();
      return proc;
    },
    defaultParams: {
      filterLength: { value: 512, min: 128, max: 2048, step: 64, label: 'NLMS Filter Length' },
      stepSize: { value: 0.3, min: 0.05, max: 0.8, step: 0.05, label: 'NLMS Step Size' }
    },
    async: true
  },
  spectralVAD: {
    name: 'Spectral Gate + VAD',
    create: (params) => new SpectralGateVAD(params.fftSize, params.voiceMinFreq, params.voiceMaxFreq),
    defaultParams: {
      fftSize: { value: 2048, min: 512, max: 4096, step: 512, label: 'FFT Size' },
      voiceMinFreq: { value: 80, min: 50, max: 200, step: 10, label: 'Voice Min Hz' },
      voiceMaxFreq: { value: 3500, min: 2000, max: 8000, step: 100, label: 'Voice Max Hz' }
    }
  },
  hpss: {
    name: 'HPSS (Harmonic Sep)',
    create: (params) => new HPSSProcessor(params.fftSize, params.harmonicKernel, params.percussiveKernel),
    defaultParams: {
      fftSize: { value: 2048, min: 1024, max: 4096, step: 512, label: 'FFT Size' },
      harmonicKernel: { value: 17, min: 5, max: 31, step: 2, label: 'Harmonic Kernel' },
      percussiveKernel: { value: 17, min: 5, max: 31, step: 2, label: 'Percussive Kernel' }
    }
  }
};

console.log('Deep Learning algorithms loaded. RNNoise WASM will be loaded on first use.');

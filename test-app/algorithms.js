/**
 * Audio Separation Algorithms
 * Pure JavaScript implementations for comparing different approaches
 */

// ============================================================
// NLMS - Normalized Least Mean Squares
// ============================================================
class NLMSFilter {
  constructor(filterLength = 1024, stepSize = 0.5, eps = 1e-6) {
    this.filterLength = filterLength;
    this.mu = stepSize;
    this.eps = eps;
    this.weights = new Float32Array(filterLength);
    this.buffer = new Float32Array(filterLength);
    this.bufferIndex = 0;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      // Add reference sample to buffer
      this.buffer[this.bufferIndex] = reference[i];
      
      // Calculate filter output (estimated echo)
      let y = 0;
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIndex - j + this.filterLength) % this.filterLength;
        y += this.weights[j] * this.buffer[idx];
      }
      
      // Error signal (desired output)
      const error = input[i] - y;
      output[i] = error;
      
      // Calculate normalization factor
      let norm = this.eps;
      for (let j = 0; j < this.filterLength; j++) {
        norm += this.buffer[j] * this.buffer[j];
      }
      
      // Update weights
      const stepNorm = this.mu / norm;
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIndex - j + this.filterLength) % this.filterLength;
        this.weights[j] += stepNorm * error * this.buffer[idx];
      }
      
      this.bufferIndex = (this.bufferIndex + 1) % this.filterLength;
    }
    
    return output;
  }
}

// ============================================================
// RLS - Recursive Least Squares
// ============================================================
class RLSFilter {
  constructor(filterLength = 512, forgettingFactor = 0.999, delta = 0.01) {
    this.filterLength = filterLength;
    this.lambda = forgettingFactor;
    this.delta = delta;
    
    this.weights = new Float32Array(filterLength);
    this.buffer = new Float32Array(filterLength);
    this.bufferIndex = 0;
    
    // Initialize P matrix (inverse correlation matrix) as diagonal
    this.P = [];
    for (let i = 0; i < filterLength; i++) {
      this.P[i] = new Float32Array(filterLength);
      this.P[i][i] = 1 / delta;
    }
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const n = this.filterLength;
    
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.bufferIndex] = reference[i];
      
      // Get current input vector x
      const x = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        x[j] = this.buffer[(this.bufferIndex - j + n) % n];
      }
      
      // Calculate filter output
      let y = 0;
      for (let j = 0; j < n; j++) {
        y += this.weights[j] * x[j];
      }
      
      // Error
      const error = input[i] - y;
      output[i] = error;
      
      // Calculate gain vector k = P*x / (lambda + x'*P*x)
      const Px = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          Px[j] += this.P[j][k] * x[k];
        }
      }
      
      let xPx = 0;
      for (let j = 0; j < n; j++) {
        xPx += x[j] * Px[j];
      }
      
      const denom = this.lambda + xPx;
      const gain = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        gain[j] = Px[j] / denom;
      }
      
      // Update weights
      for (let j = 0; j < n; j++) {
        this.weights[j] += gain[j] * error;
      }
      
      // Update P matrix: P = (P - k*x'*P) / lambda
      // Simplified update for efficiency
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          this.P[j][k] = (this.P[j][k] - gain[j] * Px[k]) / this.lambda;
        }
      }
      
      this.bufferIndex = (this.bufferIndex + 1) % n;
    }
    
    return output;
  }
}

// ============================================================
// Spectral Subtraction
// ============================================================
class SpectralSubtraction {
  constructor(fftSize = 2048, overSubtraction = 2.0, floorFactor = 0.01) {
    this.fftSize = fftSize;
    this.hopSize = fftSize / 4;
    this.alpha = overSubtraction;
    this.beta = floorFactor;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const fftSize = this.fftSize;
    const hopSize = this.hopSize;
    
    // Hann window
    const window = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));
    }
    
    // Process in overlapping frames
    const numFrames = Math.floor((input.length - fftSize) / hopSize) + 1;
    
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopSize;
      
      // Extract and window frames
      const inputFrame = new Float32Array(fftSize);
      const refFrame = new Float32Array(fftSize);
      
      for (let i = 0; i < fftSize && start + i < input.length; i++) {
        inputFrame[i] = input[start + i] * window[i];
        refFrame[i] = reference[start + i] * window[i];
      }
      
      // FFT (simple DFT for demo - in production use FFT library)
      const inputSpec = this.dft(inputFrame);
      const refSpec = this.dft(refFrame);
      
      // Spectral subtraction
      const outputSpec = new Float32Array(fftSize * 2);
      for (let k = 0; k < fftSize; k++) {
        const inputMag = Math.sqrt(inputSpec[k*2]**2 + inputSpec[k*2+1]**2);
        const refMag = Math.sqrt(refSpec[k*2]**2 + refSpec[k*2+1]**2);
        const inputPhase = Math.atan2(inputSpec[k*2+1], inputSpec[k*2]);
        
        // Subtract reference magnitude
        let outputMag = inputMag - this.alpha * refMag;
        outputMag = Math.max(outputMag, this.beta * inputMag); // Spectral floor
        
        outputSpec[k*2] = outputMag * Math.cos(inputPhase);
        outputSpec[k*2+1] = outputMag * Math.sin(inputPhase);
      }
      
      // IDFT
      const outputFrame = this.idft(outputSpec);
      
      // Overlap-add
      for (let i = 0; i < fftSize && start + i < output.length; i++) {
        output[start + i] += outputFrame[i] * window[i];
      }
    }
    
    // Normalize
    const maxVal = Math.max(...output.map(Math.abs));
    if (maxVal > 0) {
      for (let i = 0; i < output.length; i++) {
        output[i] /= maxVal;
      }
    }
    
    return output;
  }

  dft(x) {
    const N = x.length;
    const X = new Float32Array(N * 2);
    
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        re += x[n] * Math.cos(angle);
        im += x[n] * Math.sin(angle);
      }
      X[k*2] = re;
      X[k*2+1] = im;
    }
    
    return X;
  }

  idft(X) {
    const N = X.length / 2;
    const x = new Float32Array(N);
    
    for (let n = 0; n < N; n++) {
      let re = 0;
      for (let k = 0; k < N; k++) {
        const angle = 2 * Math.PI * k * n / N;
        re += X[k*2] * Math.cos(angle) - X[k*2+1] * Math.sin(angle);
      }
      x[n] = re / N;
    }
    
    return x;
  }
}

// ============================================================
// Wiener Filter (Frequency Domain)
// ============================================================
class WienerFilter {
  constructor(fftSize = 2048, noiseEstFrames = 10) {
    this.fftSize = fftSize;
    this.hopSize = fftSize / 4;
    this.noiseEstFrames = noiseEstFrames;
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
    
    // Estimate reference power spectrum
    const refPower = new Float32Array(fftSize);
    let refFrameCount = 0;
    
    for (let frame = 0; frame < Math.min(numFrames, this.noiseEstFrames); frame++) {
      const start = frame * hopSize;
      const refFrame = new Float32Array(fftSize);
      
      for (let i = 0; i < fftSize && start + i < reference.length; i++) {
        refFrame[i] = reference[start + i] * window[i];
      }
      
      const refSpec = this.dft(refFrame);
      
      for (let k = 0; k < fftSize; k++) {
        refPower[k] += refSpec[k*2]**2 + refSpec[k*2+1]**2;
      }
      refFrameCount++;
    }
    
    for (let k = 0; k < fftSize; k++) {
      refPower[k] /= refFrameCount;
    }
    
    // Process frames with Wiener filter
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopSize;
      
      const inputFrame = new Float32Array(fftSize);
      for (let i = 0; i < fftSize && start + i < input.length; i++) {
        inputFrame[i] = input[start + i] * window[i];
      }
      
      const inputSpec = this.dft(inputFrame);
      const outputSpec = new Float32Array(fftSize * 2);
      
      for (let k = 0; k < fftSize; k++) {
        const inputPower = inputSpec[k*2]**2 + inputSpec[k*2+1]**2;
        
        // Wiener gain: H = max(0, 1 - noise/signal)
        const gain = Math.max(0, 1 - refPower[k] / (inputPower + 1e-10));
        
        outputSpec[k*2] = inputSpec[k*2] * gain;
        outputSpec[k*2+1] = inputSpec[k*2+1] * gain;
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
    const X = new Float32Array(N * 2);
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        re += x[n] * Math.cos(angle);
        im += x[n] * Math.sin(angle);
      }
      X[k*2] = re;
      X[k*2+1] = im;
    }
    return X;
  }

  idft(X) {
    const N = X.length / 2;
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      let re = 0;
      for (let k = 0; k < N; k++) {
        const angle = 2 * Math.PI * k * n / N;
        re += X[k*2] * Math.cos(angle) - X[k*2+1] * Math.sin(angle);
      }
      x[n] = re / N;
    }
    return x;
  }
}

// ============================================================
// Griffiths-Jim Beamformer (Simplified)
// ============================================================
class GriffithsJim {
  constructor(filterLength = 512, stepSize = 0.1) {
    this.filterLength = filterLength;
    this.mu = stepSize;
    this.weights = new Float32Array(filterLength);
    this.buffer = new Float32Array(filterLength);
    this.bufferIndex = 0;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      // Blocking matrix output (difference signal)
      const blocked = input[i] - reference[i];
      
      // Add to adaptive filter buffer
      this.buffer[this.bufferIndex] = blocked;
      
      // Calculate adaptive filter output
      let y = 0;
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIndex - j + this.filterLength) % this.filterLength;
        y += this.weights[j] * this.buffer[idx];
      }
      
      // Output is input minus adaptive filter output
      const error = input[i] - y;
      output[i] = error;
      
      // LMS update
      let norm = 1e-6;
      for (let j = 0; j < this.filterLength; j++) {
        norm += this.buffer[j] * this.buffer[j];
      }
      
      const stepNorm = this.mu / norm;
      for (let j = 0; j < this.filterLength; j++) {
        const idx = (this.bufferIndex - j + this.filterLength) % this.filterLength;
        this.weights[j] += stepNorm * error * this.buffer[idx];
      }
      
      this.bufferIndex = (this.bufferIndex + 1) % this.filterLength;
    }
    
    return output;
  }
}

// ============================================================
// Kalman Filter for Echo Cancellation
// ============================================================
class KalmanFilter {
  constructor(filterLength = 256, processNoise = 0.001, measurementNoise = 0.1) {
    this.filterLength = filterLength;
    this.Q = processNoise;
    this.R = measurementNoise;
    
    this.weights = new Float32Array(filterLength);
    this.P = new Float32Array(filterLength).fill(1); // Diagonal covariance
    this.buffer = new Float32Array(filterLength);
    this.bufferIndex = 0;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const n = this.filterLength;
    
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.bufferIndex] = reference[i];
      
      // Prediction step (state transition is identity)
      for (let j = 0; j < n; j++) {
        this.P[j] += this.Q;
      }
      
      // Calculate predicted output
      let y = 0;
      for (let j = 0; j < n; j++) {
        const idx = (this.bufferIndex - j + n) % n;
        y += this.weights[j] * this.buffer[idx];
      }
      
      // Innovation (measurement residual)
      const innovation = input[i] - y;
      output[i] = innovation;
      
      // Kalman gain and update
      for (let j = 0; j < n; j++) {
        const idx = (this.bufferIndex - j + n) % n;
        const x = this.buffer[idx];
        
        // Innovation covariance
        const S = this.P[j] * x * x + this.R;
        
        // Kalman gain
        const K = this.P[j] * x / S;
        
        // Update state
        this.weights[j] += K * innovation;
        
        // Update covariance
        this.P[j] = (1 - K * x) * this.P[j];
      }
      
      this.bufferIndex = (this.bufferIndex + 1) % n;
    }
    
    return output;
  }
}

// ============================================================
// Algorithm Registry
// ============================================================
const algorithms = {
  nlms: {
    name: 'NLMS',
    create: (params) => new NLMSFilter(params.filterLength, params.stepSize, params.eps),
    defaultParams: {
      filterLength: { value: 1024, min: 64, max: 4096, step: 64, label: 'Filter Length' },
      stepSize: { value: 0.5, min: 0.01, max: 1.0, step: 0.01, label: 'Step Size (μ)' },
      eps: { value: 1e-6, min: 1e-8, max: 1e-4, step: 1e-8, label: 'Regularization (ε)' }
    }
  },
  rls: {
    name: 'RLS',
    create: (params) => new RLSFilter(params.filterLength, params.forgettingFactor, params.delta),
    defaultParams: {
      filterLength: { value: 256, min: 32, max: 1024, step: 32, label: 'Filter Length' },
      forgettingFactor: { value: 0.999, min: 0.9, max: 0.9999, step: 0.001, label: 'Forgetting Factor (λ)' },
      delta: { value: 0.01, min: 0.001, max: 1, step: 0.001, label: 'Initialization (δ)' }
    }
  },
  spectral: {
    name: 'Spectral Subtraction',
    create: (params) => new SpectralSubtraction(params.fftSize, params.overSubtraction, params.floorFactor),
    defaultParams: {
      fftSize: { value: 1024, min: 256, max: 4096, step: 256, label: 'FFT Size' },
      overSubtraction: { value: 2.0, min: 0.5, max: 5.0, step: 0.1, label: 'Over-subtraction (α)' },
      floorFactor: { value: 0.01, min: 0.001, max: 0.1, step: 0.001, label: 'Spectral Floor (β)' }
    }
  },
  wiener: {
    name: 'Wiener Filter',
    create: (params) => new WienerFilter(params.fftSize, params.noiseEstFrames),
    defaultParams: {
      fftSize: { value: 1024, min: 256, max: 4096, step: 256, label: 'FFT Size' },
      noiseEstFrames: { value: 10, min: 1, max: 50, step: 1, label: 'Noise Est. Frames' }
    }
  },
  griffiths: {
    name: 'Griffiths-Jim',
    create: (params) => new GriffithsJim(params.filterLength, params.stepSize),
    defaultParams: {
      filterLength: { value: 512, min: 64, max: 2048, step: 64, label: 'Filter Length' },
      stepSize: { value: 0.1, min: 0.01, max: 0.5, step: 0.01, label: 'Step Size (μ)' }
    }
  },
  kalman: {
    name: 'Kalman Filter',
    create: (params) => new KalmanFilter(params.filterLength, params.processNoise, params.measurementNoise),
    defaultParams: {
      filterLength: { value: 256, min: 32, max: 1024, step: 32, label: 'Filter Length' },
      processNoise: { value: 0.001, min: 0.0001, max: 0.1, step: 0.0001, label: 'Process Noise (Q)' },
      measurementNoise: { value: 0.1, min: 0.01, max: 1.0, step: 0.01, label: 'Measurement Noise (R)' }
    }
  }
};

// Export for use in app.js
window.algorithms = algorithms;


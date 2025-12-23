/**
 * Advanced Audio Separation Algorithms
 * More sophisticated signal processing techniques
 */

// ============================================================
// Frequency-Domain Adaptive Filter (FDAF)
// ============================================================
class FDAFFilter {
  constructor(blockSize = 512, stepSize = 0.5) {
    this.blockSize = blockSize;
    this.mu = stepSize;
    this.fftSize = blockSize * 2;
    
    // Frequency domain weights (complex)
    this.W_re = new Float32Array(this.fftSize);
    this.W_im = new Float32Array(this.fftSize);
    
    // Buffers
    this.inputBuffer = new Float32Array(this.fftSize);
    this.refBuffer = new Float32Array(this.fftSize);
    this.outputBuffer = new Float32Array(blockSize);
    this.bufferPos = 0;
  }

  fft(real, imag) {
    const n = real.length;
    if (n <= 1) return;
    
    // Bit reversal
    for (let i = 0, j = 0; i < n; i++) {
      if (j > i) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
      let m = n >> 1;
      while (m >= 1 && j >= m) { j -= m; m >>= 1; }
      j += m;
    }
    
    // FFT
    for (let mmax = 1; mmax < n; mmax <<= 1) {
      const theta = -Math.PI / mmax;
      const wpr = Math.cos(theta), wpi = Math.sin(theta);
      let wr = 1, wi = 0;
      
      for (let m = 0; m < mmax; m++) {
        for (let i = m; i < n; i += mmax << 1) {
          const j = i + mmax;
          const tr = wr * real[j] - wi * imag[j];
          const ti = wr * imag[j] + wi * real[j];
          real[j] = real[i] - tr;
          imag[j] = imag[i] - ti;
          real[i] += tr;
          imag[i] += ti;
        }
        const wtemp = wr;
        wr = wr * wpr - wi * wpi;
        wi = wi * wpr + wtemp * wpi;
      }
    }
  }

  ifft(real, imag) {
    const n = real.length;
    for (let i = 0; i < n; i++) imag[i] = -imag[i];
    this.fft(real, imag);
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] = -imag[i] / n;
    }
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    let outIdx = 0;
    
    for (let i = 0; i < input.length; i++) {
      this.inputBuffer[this.bufferPos] = input[i];
      this.refBuffer[this.bufferPos] = reference[i];
      this.bufferPos++;
      
      if (this.bufferPos >= this.blockSize) {
        // Process block
        const X_re = new Float32Array(this.fftSize);
        const X_im = new Float32Array(this.fftSize);
        const R_re = new Float32Array(this.fftSize);
        const R_im = new Float32Array(this.fftSize);
        
        // Copy with zero padding
        for (let j = 0; j < this.fftSize; j++) {
          X_re[j] = j < this.blockSize ? this.inputBuffer[j] : 0;
          R_re[j] = j < this.blockSize ? this.refBuffer[j] : 0;
        }
        
        // FFT
        this.fft(X_re, X_im);
        this.fft(R_re, R_im);
        
        // Filter output Y = W * R
        const Y_re = new Float32Array(this.fftSize);
        const Y_im = new Float32Array(this.fftSize);
        for (let k = 0; k < this.fftSize; k++) {
          Y_re[k] = this.W_re[k] * R_re[k] - this.W_im[k] * R_im[k];
          Y_im[k] = this.W_re[k] * R_im[k] + this.W_im[k] * R_re[k];
        }
        
        // IFFT
        this.ifft(Y_re, Y_im);
        
        // Error = X - Y
        const E_re = new Float32Array(this.fftSize);
        const E_im = new Float32Array(this.fftSize);
        for (let j = 0; j < this.blockSize; j++) {
          E_re[j] = this.inputBuffer[j] - Y_re[j];
          output[outIdx + j] = E_re[j];
        }
        
        // FFT of error
        this.fft(E_re, E_im);
        
        // Update weights: W += mu * E * conj(R) / |R|^2
        for (let k = 0; k < this.fftSize; k++) {
          const Rpower = R_re[k] * R_re[k] + R_im[k] * R_im[k] + 1e-8;
          const dW_re = (E_re[k] * R_re[k] + E_im[k] * R_im[k]) / Rpower;
          const dW_im = (E_im[k] * R_re[k] - E_re[k] * R_im[k]) / Rpower;
          this.W_re[k] += this.mu * dW_re;
          this.W_im[k] += this.mu * dW_im;
        }
        
        outIdx += this.blockSize;
        
        // Shift buffer
        this.bufferPos = 0;
      }
    }
    
    return output;
  }
}

// ============================================================
// Subband Adaptive Filter
// ============================================================
class SubbandFilter {
  constructor(numBands = 8, filterLength = 64, stepSize = 0.3) {
    this.numBands = numBands;
    this.filterLength = filterLength;
    this.mu = stepSize;
    
    // Per-band filters
    this.weights = [];
    this.buffers = [];
    for (let b = 0; b < numBands; b++) {
      this.weights[b] = new Float32Array(filterLength);
      this.buffers[b] = { input: new Float32Array(filterLength), ref: new Float32Array(filterLength), idx: 0 };
    }
    
    // Analysis/synthesis filter bank (simple DFT-based)
    this.fftSize = numBands * 2;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    const hopSize = this.numBands;
    const frameBuffer = new Float32Array(this.fftSize);
    const refFrameBuffer = new Float32Array(this.fftSize);
    
    for (let i = 0; i < input.length; i += hopSize) {
      // Fill frame
      for (let j = 0; j < this.fftSize && i + j < input.length; j++) {
        frameBuffer[j] = input[i + j];
        refFrameBuffer[j] = reference[i + j];
      }
      
      // Simple band decomposition
      const inputBands = new Float32Array(this.numBands);
      const refBands = new Float32Array(this.numBands);
      
      for (let b = 0; b < this.numBands; b++) {
        const freq = (b + 0.5) * Math.PI / this.numBands;
        for (let j = 0; j < this.fftSize; j++) {
          inputBands[b] += frameBuffer[j] * Math.cos(freq * j);
          refBands[b] += refFrameBuffer[j] * Math.cos(freq * j);
        }
        inputBands[b] /= this.fftSize;
        refBands[b] /= this.fftSize;
      }
      
      // Process each band with NLMS
      const outputBands = new Float32Array(this.numBands);
      for (let b = 0; b < this.numBands; b++) {
        const buf = this.buffers[b];
        buf.ref[buf.idx] = refBands[b];
        
        // Filter output
        let y = 0;
        for (let j = 0; j < this.filterLength; j++) {
          const idx = (buf.idx - j + this.filterLength) % this.filterLength;
          y += this.weights[b][j] * buf.ref[idx];
        }
        
        const error = inputBands[b] - y;
        outputBands[b] = error;
        
        // Update
        let norm = 1e-8;
        for (let j = 0; j < this.filterLength; j++) norm += buf.ref[j] ** 2;
        
        for (let j = 0; j < this.filterLength; j++) {
          const idx = (buf.idx - j + this.filterLength) % this.filterLength;
          this.weights[b][j] += this.mu / norm * error * buf.ref[idx];
        }
        
        buf.idx = (buf.idx + 1) % this.filterLength;
      }
      
      // Synthesis
      for (let j = 0; j < hopSize && i + j < output.length; j++) {
        let sample = 0;
        for (let b = 0; b < this.numBands; b++) {
          const freq = (b + 0.5) * Math.PI / this.numBands;
          sample += outputBands[b] * Math.cos(freq * j) * 2;
        }
        output[i + j] = sample;
      }
    }
    
    return output;
  }
}

// ============================================================
// Affine Projection Algorithm (APA)
// ============================================================
class APAFilter {
  constructor(filterLength = 512, projectionOrder = 4, stepSize = 0.5, reg = 0.01) {
    this.filterLength = filterLength;
    this.P = projectionOrder;
    this.mu = stepSize;
    this.delta = reg;
    
    this.weights = new Float32Array(filterLength);
    this.buffer = new Float32Array(filterLength);
    this.bufferIdx = 0;
    
    // Store last P input vectors and errors
    this.inputHistory = [];
    this.errorHistory = [];
    for (let i = 0; i < projectionOrder; i++) {
      this.inputHistory.push(new Float32Array(filterLength));
      this.errorHistory.push(0);
    }
    this.historyIdx = 0;
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.bufferIdx] = reference[i];
      
      // Store current input vector
      const currentInput = new Float32Array(this.filterLength);
      for (let j = 0; j < this.filterLength; j++) {
        currentInput[j] = this.buffer[(this.bufferIdx - j + this.filterLength) % this.filterLength];
      }
      
      // Filter output
      let y = 0;
      for (let j = 0; j < this.filterLength; j++) {
        y += this.weights[j] * currentInput[j];
      }
      
      const error = input[i] - y;
      output[i] = error;
      
      // Store in history
      this.inputHistory[this.historyIdx] = currentInput;
      this.errorHistory[this.historyIdx] = error;
      
      // APA update using last P samples
      // Simplified: average gradient from P directions
      const gradient = new Float32Array(this.filterLength);
      let totalNorm = this.delta;
      
      for (let p = 0; p < this.P; p++) {
        const idx = (this.historyIdx - p + this.P) % this.P;
        const x = this.inputHistory[idx];
        const e = this.errorHistory[idx];
        
        let norm = 0;
        for (let j = 0; j < this.filterLength; j++) norm += x[j] ** 2;
        totalNorm += norm;
        
        for (let j = 0; j < this.filterLength; j++) {
          gradient[j] += e * x[j];
        }
      }
      
      // Update weights
      for (let j = 0; j < this.filterLength; j++) {
        this.weights[j] += this.mu * gradient[j] / totalNorm;
      }
      
      this.historyIdx = (this.historyIdx + 1) % this.P;
      this.bufferIdx = (this.bufferIdx + 1) % this.filterLength;
    }
    
    return output;
  }
}

// ============================================================
// Partitioned Block Frequency Domain (PBFDAF)
// ============================================================
class PBFDAFFilter {
  constructor(blockSize = 256, numPartitions = 4, stepSize = 0.5) {
    this.blockSize = blockSize;
    this.numPartitions = numPartitions;
    this.mu = stepSize;
    this.fftSize = blockSize * 2;
    
    // Partitioned weights
    this.W_re = [];
    this.W_im = [];
    for (let p = 0; p < numPartitions; p++) {
      this.W_re.push(new Float32Array(this.fftSize));
      this.W_im.push(new Float32Array(this.fftSize));
    }
    
    // Reference buffer for each partition
    this.refBuffers_re = [];
    this.refBuffers_im = [];
    for (let p = 0; p < numPartitions; p++) {
      this.refBuffers_re.push(new Float32Array(this.fftSize));
      this.refBuffers_im.push(new Float32Array(this.fftSize));
    }
    
    this.inputBuffer = new Float32Array(blockSize);
    this.refBuffer = new Float32Array(blockSize);
    this.bufferPos = 0;
  }

  fft(real, imag) {
    const n = real.length;
    for (let i = 0, j = 0; i < n; i++) {
      if (j > i) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
      let m = n >> 1;
      while (m >= 1 && j >= m) { j -= m; m >>= 1; }
      j += m;
    }
    for (let mmax = 1; mmax < n; mmax <<= 1) {
      const theta = -Math.PI / mmax;
      let wr = 1, wi = 0;
      const wpr = Math.cos(theta), wpi = Math.sin(theta);
      for (let m = 0; m < mmax; m++) {
        for (let i = m; i < n; i += mmax << 1) {
          const j = i + mmax;
          const tr = wr * real[j] - wi * imag[j];
          const ti = wr * imag[j] + wi * real[j];
          real[j] = real[i] - tr; imag[j] = imag[i] - ti;
          real[i] += tr; imag[i] += ti;
        }
        const wt = wr; wr = wr * wpr - wi * wpi; wi = wi * wpr + wt * wpi;
      }
    }
  }

  ifft(real, imag) {
    const n = real.length;
    for (let i = 0; i < n; i++) imag[i] = -imag[i];
    this.fft(real, imag);
    for (let i = 0; i < n; i++) { real[i] /= n; imag[i] = -imag[i] / n; }
  }

  process(input, reference) {
    const output = new Float32Array(input.length);
    let outIdx = 0;
    
    for (let i = 0; i < input.length; i++) {
      this.inputBuffer[this.bufferPos] = input[i];
      this.refBuffer[this.bufferPos] = reference[i];
      this.bufferPos++;
      
      if (this.bufferPos >= this.blockSize) {
        // Shift reference partitions
        for (let p = this.numPartitions - 1; p > 0; p--) {
          this.refBuffers_re[p] = this.refBuffers_re[p - 1].slice();
          this.refBuffers_im[p] = this.refBuffers_im[p - 1].slice();
        }
        
        // New reference FFT
        const R_re = new Float32Array(this.fftSize);
        const R_im = new Float32Array(this.fftSize);
        for (let j = 0; j < this.blockSize; j++) R_re[j] = this.refBuffer[j];
        this.fft(R_re, R_im);
        this.refBuffers_re[0] = R_re;
        this.refBuffers_im[0] = R_im;
        
        // Sum filter outputs from all partitions
        const Y_re = new Float32Array(this.fftSize);
        const Y_im = new Float32Array(this.fftSize);
        
        for (let p = 0; p < this.numPartitions; p++) {
          for (let k = 0; k < this.fftSize; k++) {
            Y_re[k] += this.W_re[p][k] * this.refBuffers_re[p][k] - this.W_im[p][k] * this.refBuffers_im[p][k];
            Y_im[k] += this.W_re[p][k] * this.refBuffers_im[p][k] + this.W_im[p][k] * this.refBuffers_re[p][k];
          }
        }
        
        this.ifft(Y_re, Y_im);
        
        // Error
        for (let j = 0; j < this.blockSize; j++) {
          output[outIdx + j] = this.inputBuffer[j] - Y_re[j];
        }
        
        // Update all partitions
        const E_re = new Float32Array(this.fftSize);
        const E_im = new Float32Array(this.fftSize);
        for (let j = 0; j < this.blockSize; j++) E_re[j] = output[outIdx + j];
        this.fft(E_re, E_im);
        
        for (let p = 0; p < this.numPartitions; p++) {
          for (let k = 0; k < this.fftSize; k++) {
            const Rp = this.refBuffers_re[p][k] ** 2 + this.refBuffers_im[p][k] ** 2 + 1e-8;
            this.W_re[p][k] += this.mu * (E_re[k] * this.refBuffers_re[p][k] + E_im[k] * this.refBuffers_im[p][k]) / Rp;
            this.W_im[p][k] += this.mu * (E_im[k] * this.refBuffers_re[p][k] - E_re[k] * this.refBuffers_im[p][k]) / Rp;
          }
        }
        
        outIdx += this.blockSize;
        this.bufferPos = 0;
      }
    }
    
    return output;
  }
}

// ============================================================
// Independent Component Analysis (FastICA simplified)
// ============================================================
class ICAFilter {
  constructor(maxIterations = 100, tolerance = 1e-4) {
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  process(input, reference) {
    const n = Math.min(input.length, reference.length);
    const output = new Float32Array(n);
    
    // Center the data
    let meanInput = 0, meanRef = 0;
    for (let i = 0; i < n; i++) {
      meanInput += input[i];
      meanRef += reference[i];
    }
    meanInput /= n;
    meanRef /= n;
    
    const X = new Float32Array(n);
    const R = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      X[i] = input[i] - meanInput;
      R[i] = reference[i] - meanRef;
    }
    
    // Whitening (simplified)
    let varX = 0, varR = 0, covXR = 0;
    for (let i = 0; i < n; i++) {
      varX += X[i] ** 2;
      varR += R[i] ** 2;
      covXR += X[i] * R[i];
    }
    varX /= n; varR /= n; covXR /= n;
    
    // FastICA iteration to find unmixing vector
    let w1 = 1, w2 = -covXR / (varR + 1e-8);
    const norm = Math.sqrt(w1 ** 2 + w2 ** 2);
    w1 /= norm; w2 /= norm;
    
    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Compute y = w1*X + w2*R
      let sum_g = 0, sum_gp = 0;
      for (let i = 0; i < n; i++) {
        const y = w1 * X[i] + w2 * R[i];
        const g = Math.tanh(y);
        const gp = 1 - g ** 2;
        sum_g += X[i] * g;
        sum_gp += gp;
      }
      
      // Update
      const w1_new = sum_g / n - (sum_gp / n) * w1;
      
      sum_g = 0;
      for (let i = 0; i < n; i++) {
        const y = w1 * X[i] + w2 * R[i];
        sum_g += R[i] * Math.tanh(y);
      }
      const w2_new = sum_g / n - (sum_gp / n) * w2;
      
      // Normalize
      const newNorm = Math.sqrt(w1_new ** 2 + w2_new ** 2);
      w1 = w1_new / newNorm;
      w2 = w2_new / newNorm;
    }
    
    // Apply unmixing - we want the component that's mostly input (voice)
    // If w2 is large, the reference is being mixed in, so we want to minimize that
    if (Math.abs(w2) > Math.abs(w1)) {
      // Swap to get voice-dominant component
      [w1, w2] = [w2, w1];
    }
    
    for (let i = 0; i < n; i++) {
      output[i] = w1 * X[i] + w2 * R[i];
    }
    
    // Normalize output
    let maxAbs = 0;
    for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(output[i]));
    if (maxAbs > 0) {
      for (let i = 0; i < n; i++) output[i] /= maxAbs;
    }
    
    return output;
  }
}

// Export
window.advancedAlgorithms = {
  fdaf: {
    name: 'FDAF (Freq Domain)',
    create: (params) => new FDAFFilter(params.blockSize, params.stepSize),
    defaultParams: {
      blockSize: { value: 512, min: 128, max: 2048, step: 128, label: 'Block Size' },
      stepSize: { value: 0.5, min: 0.1, max: 1.0, step: 0.05, label: 'Step Size' }
    }
  },
  subband: {
    name: 'Subband Filter',
    create: (params) => new SubbandFilter(params.numBands, params.filterLength, params.stepSize),
    defaultParams: {
      numBands: { value: 8, min: 4, max: 32, step: 4, label: 'Num Bands' },
      filterLength: { value: 64, min: 16, max: 256, step: 16, label: 'Filter Length' },
      stepSize: { value: 0.3, min: 0.05, max: 0.8, step: 0.05, label: 'Step Size' }
    }
  },
  apa: {
    name: 'APA',
    create: (params) => new APAFilter(params.filterLength, params.projectionOrder, params.stepSize),
    defaultParams: {
      filterLength: { value: 512, min: 64, max: 2048, step: 64, label: 'Filter Length' },
      projectionOrder: { value: 4, min: 2, max: 16, step: 1, label: 'Projection Order' },
      stepSize: { value: 0.5, min: 0.1, max: 1.0, step: 0.05, label: 'Step Size' }
    }
  },
  pbfdaf: {
    name: 'PBFDAF',
    create: (params) => new PBFDAFFilter(params.blockSize, params.numPartitions, params.stepSize),
    defaultParams: {
      blockSize: { value: 256, min: 64, max: 1024, step: 64, label: 'Block Size' },
      numPartitions: { value: 4, min: 2, max: 16, step: 1, label: 'Partitions' },
      stepSize: { value: 0.5, min: 0.1, max: 1.0, step: 0.05, label: 'Step Size' }
    }
  },
  ica: {
    name: 'ICA (FastICA)',
    create: (params) => new ICAFilter(params.maxIterations, params.tolerance),
    defaultParams: {
      maxIterations: { value: 100, min: 10, max: 500, step: 10, label: 'Max Iterations' },
      tolerance: { value: 0.0001, min: 0.00001, max: 0.01, step: 0.00001, label: 'Tolerance' }
    }
  }
};


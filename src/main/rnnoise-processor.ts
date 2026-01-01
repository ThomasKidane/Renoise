/**
 * RNNoise Processor for Real-Time Noise Suppression
 * 
 * Uses the @shiguredo/rnnoise-wasm package for high-quality
 * deep learning-based noise suppression.
 */

import { EventEmitter } from 'events';
import { Rnnoise, DenoiseState } from '@shiguredo/rnnoise-wasm';

// RNNoise constants
const RNNOISE_SAMPLE_RATE = 48000;
const RNNOISE_FRAME_SIZE = 480; // 10ms at 48kHz - this is what the library expects

export class RNNoiseProcessor extends EventEmitter {
  private rnnoise: Rnnoise | null = null;
  private denoiseState: DenoiseState | null = null;
  private isInitialized: boolean = false;
  private frameSize: number = RNNOISE_FRAME_SIZE;

  constructor() {
    super();
  }

  /**
   * Initialize the RNNoise WASM module
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;

    try {
      console.log('[RNNoise] Loading WASM module...');
      
      // Load the RNNoise WASM module
      this.rnnoise = await Rnnoise.load();
      this.frameSize = this.rnnoise.frameSize;
      
      console.log(`[RNNoise] Frame size: ${this.frameSize} samples`);
      
      // Create denoiser state
      this.denoiseState = this.rnnoise.createDenoiseState();
      
      this.isInitialized = true;
      console.log('[RNNoise] Initialized successfully');
      return true;
    } catch (e) {
      console.error('[RNNoise] Failed to initialize:', e);
      return false;
    }
  }

  /**
   * Process a single frame of audio
   * RNNoise expects 16-bit PCM range [-32768, 32767] in Float32Array
   * The processFrame method modifies the input IN PLACE
   */
  processFrame(input: Float32Array): Float32Array {
    if (!this.isInitialized || !this.denoiseState) {
      return input;
    }

    try {
      // Create a copy for processing (since RNNoise modifies in place)
      const frame = new Float32Array(this.frameSize);
      
      // Convert from [-1, 1] float to 16-bit PCM range [-32768, 32767]
      for (let i = 0; i < this.frameSize; i++) {
        frame[i] = (input[i] || 0) * 32767;
      }

      // Process through RNNoise - this modifies frame IN PLACE
      // Returns VAD probability (0-1), we don't use it here
      this.denoiseState.processFrame(frame);

      // Convert back from 16-bit PCM range to [-1, 1] float
      const result = new Float32Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) {
        result[i] = frame[i] / 32767;
      }

      return result;
    } catch (e) {
      console.error('[RNNoise] Processing error:', e);
      return input;
    }
  }

  /**
   * Process audio with reference signal (echo cancellation + noise suppression)
   * First subtracts the reference, then applies RNNoise
   */
  process(mic: Float32Array, ref: Float32Array): Float32Array {
    if (!this.isInitialized || !this.denoiseState) {
      console.warn('[RNNoise] Not initialized, returning mic input');
      return mic;
    }

    const output = new Float32Array(mic.length);
    const numFrames = Math.floor(mic.length / this.frameSize);

    // Process complete frames
    for (let i = 0; i < numFrames; i++) {
      const start = i * this.frameSize;
      const frame = new Float32Array(this.frameSize);
      
      // Subtract reference signal (simple echo cancellation)
      // The reference is the system audio that's bleeding into the mic
      for (let j = 0; j < this.frameSize; j++) {
        const micSample = mic[start + j] || 0;
        const refSample = ref[start + j] || 0;
        // Subtract reference at full strength - this removes the echo/music
        // RNNoise will clean up any artifacts
        frame[j] = micSample - refSample;
      }

      // Process through RNNoise to clean up residual noise
      const processed = this.processFrame(frame);

      // Copy to output
      for (let j = 0; j < this.frameSize; j++) {
        output[start + j] = processed[j];
      }
    }

    // Handle remaining samples (pad to frame size)
    const remaining = mic.length % this.frameSize;
    if (remaining > 0) {
      const start = numFrames * this.frameSize;
      const frame = new Float32Array(this.frameSize);
      
      for (let j = 0; j < remaining; j++) {
        const micSample = mic[start + j] || 0;
        const refSample = ref[start + j] || 0;
        frame[j] = micSample - refSample;
      }
      // Rest of frame is zeros (already initialized)

      const processed = this.processFrame(frame);

      for (let j = 0; j < remaining; j++) {
        output[start + j] = processed[j];
      }
    }

    return output;
  }

  /**
   * Reset the processor state
   */
  reset(): void {
    if (this.rnnoise && this.denoiseState) {
      this.denoiseState.destroy();
      this.denoiseState = this.rnnoise.createDenoiseState();
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.denoiseState) {
      this.denoiseState.destroy();
      this.denoiseState = null;
    }
    this.isInitialized = false;
    this.rnnoise = null;
  }

  /**
   * Check if ready
   */
  isReady(): boolean {
    return this.isInitialized && this.denoiseState !== null;
  }

  /**
   * Get the frame size
   */
  getFrameSize(): number {
    return this.frameSize;
  }
}

// Singleton instance
let rnnoiseInstance: RNNoiseProcessor | null = null;

export function getRNNoiseProcessor(): RNNoiseProcessor {
  if (!rnnoiseInstance) {
    rnnoiseInstance = new RNNoiseProcessor();
  }
  return rnnoiseInstance;
}

export async function initializeRNNoise(): Promise<RNNoiseProcessor> {
  const processor = getRNNoiseProcessor();
  await processor.initialize();
  return processor;
}

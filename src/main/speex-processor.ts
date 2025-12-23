/**
 * SpeexDSP-based Audio Processor with Echo Cancellation
 * Fixed reference sampling with proper ring buffer synchronization
 * Also runs AEC3 post-processing on recordings for comparison
 */

import * as naudiodon from 'naudiodon2';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { DeviceManager, getDeviceManager } from './device-manager';
import { SpeexAEC } from './speex-aec';

export interface SpeexProcessorConfig {
  sampleRate: number;
  frameSize: number;
  filterLength: number;
  inputDeviceId: number | null;
  referenceDeviceId: number | null;
  outputDeviceId: number | null;
  inputGain: number;
  referenceGain: number;
  outputGain: number;
}

export interface AudioLevels {
  input: number;
  reference: number;
  output: number;
  convergence: number;
}

// Ring buffer for proper audio synchronization
class RingBuffer {
  private buffer: Float32Array;
  private writeIndex: number = 0;
  private readIndex: number = 0;
  private available: number = 0;
  private capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
  }

  write(samples: Float32Array): void {
    for (let i = 0; i < samples.length; i++) {
      this.buffer[this.writeIndex] = samples[i];
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      if (this.available < this.capacity) {
        this.available++;
      } else {
        // Buffer overflow - advance read index
        this.readIndex = (this.readIndex + 1) % this.capacity;
      }
    }
  }

  read(count: number): Float32Array {
    const result = new Float32Array(count);
    const toRead = Math.min(count, this.available);
    
    for (let i = 0; i < toRead; i++) {
      result[i] = this.buffer[this.readIndex];
      this.readIndex = (this.readIndex + 1) % this.capacity;
      this.available--;
    }
    
    // Fill remaining with zeros if not enough samples
    for (let i = toRead; i < count; i++) {
      result[i] = 0;
    }
    
    return result;
  }

  peek(count: number): Float32Array {
    const result = new Float32Array(count);
    const toPeek = Math.min(count, this.available);
    let idx = this.readIndex;
    
    for (let i = 0; i < toPeek; i++) {
      result[i] = this.buffer[idx];
      idx = (idx + 1) % this.capacity;
    }
    
    return result;
  }

  getAvailable(): number {
    return this.available;
  }

  clear(): void {
    this.writeIndex = 0;
    this.readIndex = 0;
    this.available = 0;
  }
}

const DEFAULT_CONFIG: SpeexProcessorConfig = {
  sampleRate: 48000,  // Higher sample rate for better quality
  frameSize: 480,     // 10ms at 48kHz
  filterLength: 4800, // 100ms filter length
  inputDeviceId: null,
  referenceDeviceId: null,
  outputDeviceId: null,
  inputGain: 1.0,
  referenceGain: 1.0,
  outputGain: 1.0
};

export class SpeexProcessor extends EventEmitter {
  private config: SpeexProcessorConfig;
  private deviceManager: DeviceManager;
  
  private inputStream: any = null;
  private referenceStream: any = null;
  private outputStream: any = null;
  
  private aec: SpeexAEC;
  
  // Ring buffers for proper synchronization
  private inputBuffer: RingBuffer;
  private referenceBuffer: RingBuffer;
  
  private isRunning: boolean = false;
  private levels: AudioLevels = { input: 0, reference: 0, output: 0, convergence: 0 };
  private levelLogCounter: number = 0;
  
  private isRecording: boolean = false;
  private rawInputRecording: Float32Array[] = [];
  private processedOutputRecording: Float32Array[] = [];
  private referenceRecording: Float32Array[] = [];
  
  // Processing timer for consistent frame rate
  private processTimer: NodeJS.Timeout | null = null;

  constructor(config: Partial<SpeexProcessorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.deviceManager = getDeviceManager();
    
    this.aec = new SpeexAEC({
      frameSize: this.config.frameSize,
      sampleRate: this.config.sampleRate,
      filterLength: this.config.filterLength
    });
    
    if (!this.aec.isNativeAvailable()) {
      console.warn('Native SpeexAEC not available - echo cancellation disabled');
    }
    
    // Buffer size: 100ms worth of audio (small buffer to minimize latency)
    const bufferSize = Math.floor(this.config.sampleRate * 0.1);
    this.inputBuffer = new RingBuffer(bufferSize);
    this.referenceBuffer = new RingBuffer(bufferSize);
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    console.log('Starting SpeexProcessor...');
    console.log(`Sample rate: ${this.config.sampleRate}Hz, Frame size: ${this.config.frameSize}`);
    
    try {
      await this.startStreams();
      this.isRunning = true;
      
      // Process frames frequently to keep up with real-time audio
      // Use 5ms interval to ensure we drain buffers fast enough
      this.processTimer = setInterval(() => this.processFrame(), 5);
      
      console.log('SpeexProcessor started successfully');
    } catch (error) {
      console.error('Failed to start SpeexProcessor:', error);
      this.stop();
      throw error;
    }
  }

  stop(): void {
    console.log('Stopping SpeexProcessor...');
    this.isRunning = false;
    
    if (this.processTimer) {
      clearInterval(this.processTimer);
      this.processTimer = null;
    }
    
    if (this.inputStream) {
      try { this.inputStream.quit(); } catch (e) {}
      this.inputStream = null;
    }
    if (this.referenceStream) {
      try { this.referenceStream.quit(); } catch (e) {}
      this.referenceStream = null;
    }
    if (this.outputStream) {
      try { this.outputStream.quit(); } catch (e) {}
      this.outputStream = null;
    }
    
    this.aec.reset();
    this.inputBuffer.clear();
    this.referenceBuffer.clear();
    
    console.log('SpeexProcessor stopped');
  }

  private async startStreams(): Promise<void> {
    const devices = this.deviceManager.getAllDevices();
    
    let inputDevice = devices.find(d => d.id === this.config.inputDeviceId);
    if (!inputDevice) {
      inputDevice = devices.find(d => d.maxInputChannels > 0 && d.name.includes('MacBook'));
    }
    if (!inputDevice) {
      inputDevice = devices.find(d => d.maxInputChannels > 0);
    }
    
    let referenceDevice = devices.find(d => d.id === this.config.referenceDeviceId);
    if (!referenceDevice) {
      referenceDevice = devices.find(d => d.maxInputChannels > 0 && d.name.includes('BlackHole 2ch'));
    }
    
    let outputDevice = devices.find(d => d.id === this.config.outputDeviceId);
    if (!outputDevice) {
      outputDevice = devices.find(d => d.maxOutputChannels > 0 && d.name.includes('BlackHole 16ch'));
    }
    
    if (!inputDevice) throw new Error('No input device found');
    
    console.log(`Input: ${inputDevice.name} (ID: ${inputDevice.id})`);
    if (referenceDevice) console.log(`Reference: ${referenceDevice.name} (ID: ${referenceDevice.id})`);
    if (outputDevice) console.log(`Output: ${outputDevice.name} (ID: ${outputDevice.id})`);

    // Input stream (microphone) - use higher buffer for stability
    this.inputStream = new naudiodon.AudioIO({
      inOptions: {
        channelCount: 1,
        sampleFormat: naudiodon.SampleFormatFloat32,
        sampleRate: this.config.sampleRate,
        deviceId: inputDevice.id,
        closeOnError: false,
        highwaterMark: this.config.frameSize * 4 * 4  // 4 frames buffer
      }
    });

    this.inputStream.on('data', (buffer: Buffer) => this.handleInputData(buffer));
    this.inputStream.on('error', (err: Error) => {
      console.error('Input stream error:', err);
      this.emit('error', { type: 'input', error: err });
    });
    this.inputStream.start();

    // Reference stream (system audio) - same buffer settings
    if (referenceDevice) {
      this.referenceStream = new naudiodon.AudioIO({
        inOptions: {
          channelCount: 1,
          sampleFormat: naudiodon.SampleFormatFloat32,
          sampleRate: this.config.sampleRate,
          deviceId: referenceDevice.id,
          closeOnError: false,
          highwaterMark: this.config.frameSize * 4 * 4
        }
      });

      this.referenceStream.on('data', (buffer: Buffer) => this.handleReferenceData(buffer));
      this.referenceStream.on('error', (err: Error) => {
        console.error('Reference stream error:', err);
      });
      this.referenceStream.start();
    }

    // Output stream
    if (outputDevice) {
      this.outputStream = new naudiodon.AudioIO({
        outOptions: {
          channelCount: 1,
          sampleFormat: naudiodon.SampleFormatFloat32,
          sampleRate: this.config.sampleRate,
          deviceId: outputDevice.id,
          closeOnError: false,
          highwaterMark: this.config.frameSize * 4 * 4
        }
      });

      this.outputStream.on('error', (err: Error) => {
        console.error('Output stream error:', err);
      });
      this.outputStream.start();
    }
  }

  private handleInputData(buffer: Buffer): void {
    const floatCount = buffer.length / 4;
    const samples = new Float32Array(floatCount);
    
    for (let i = 0; i < floatCount; i++) {
      samples[i] = buffer.readFloatLE(i * 4) * this.config.inputGain;
    }
    
    this.inputBuffer.write(samples);
  }

  private handleReferenceData(buffer: Buffer): void {
    const floatCount = buffer.length / 4;
    const samples = new Float32Array(floatCount);
    
    for (let i = 0; i < floatCount; i++) {
      samples[i] = buffer.readFloatLE(i * 4) * this.config.referenceGain;
    }
    
    this.referenceBuffer.write(samples);
  }

  private processFrame(): void {
    if (!this.isRunning) return;
    
    const frameSize = this.config.frameSize;
    
    // Process ALL available frames to keep up with input rate
    let framesProcessed = 0;
    const maxFramesPerTick = 50; // Limit to prevent blocking
    
    while (this.inputBuffer.getAvailable() >= frameSize && framesProcessed < maxFramesPerTick) {
      // Read input frame
      const inputFrame = this.inputBuffer.read(frameSize);
      
      // Read reference frame (use whatever is available, pad with zeros if needed)
      const referenceFrame = this.referenceBuffer.read(frameSize);
      
      const inputRMS = this.calculateRMS(inputFrame);
      const refRMS = this.calculateRMS(referenceFrame);
      
      // Process through AEC
      const output = this.aec.process(inputFrame, referenceFrame);
      
      // Apply output gain
      for (let i = 0; i < output.length; i++) {
        output[i] *= this.config.outputGain;
      }
      
      const outputRMS = this.calculateRMS(output);
      
      this.levels = {
        input: inputRMS,
        reference: refRMS,
        output: outputRMS,
        convergence: refRMS > 0.001 ? Math.min(1, outputRMS / refRMS) : 1
      };
      
      this.emit('levels', this.levels);
      
      // Recording
      if (this.isRecording) {
        this.rawInputRecording.push(new Float32Array(inputFrame));
        this.processedOutputRecording.push(new Float32Array(output));
        this.referenceRecording.push(new Float32Array(referenceFrame));
      }
      
      // Write to output
      if (this.outputStream) {
        const outputBuffer = Buffer.alloc(output.length * 4);
        for (let i = 0; i < output.length; i++) {
          outputBuffer.writeFloatLE(output[i], i * 4);
        }
        this.outputStream.write(outputBuffer);
      }
      
      framesProcessed++;
    }
    
    // Log every ~1 second (100 frames at 10ms each)
    this.levelLogCounter += framesProcessed;
    if (this.levelLogCounter >= 100) {
      console.log(`Processed ${framesProcessed} frames | Buffers: in=${this.inputBuffer.getAvailable()}, ref=${this.referenceBuffer.getAvailable()}`);
      this.levelLogCounter = 0;
    }
  }

  private calculateRMS(samples: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  getLevels(): AudioLevels { return { ...this.levels }; }
  getIsRunning(): boolean { return this.isRunning; }
  updateConfig(newConfig: Partial<SpeexProcessorConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
  getConfig(): SpeexProcessorConfig { return { ...this.config }; }

  startRecording(): void {
    if (this.isRecording) return;
    this.isRecording = true;
    this.rawInputRecording = [];
    this.processedOutputRecording = [];
    this.referenceRecording = [];
    console.log('Recording started');
  }

  async stopRecording(): Promise<{ rawInput: string; processedOutput: string; reference: string; aec3Output?: string }> {
    if (!this.isRecording) return { rawInput: '', processedOutput: '', reference: '' };
    
    this.isRecording = false;
    console.log('Recording stopped');
    
    const recordingsDir = path.join(process.cwd(), 'recordings');
    if (!fs.existsSync(recordingsDir)) {
      fs.mkdirSync(recordingsDir, { recursive: true });
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    const rawInputPath = path.join(recordingsDir, `raw_input_${timestamp}.wav`);
    await this.saveWavFile(rawInputPath, this.rawInputRecording);
    
    const processedOutputPath = path.join(recordingsDir, `processed_output_${timestamp}.wav`);
    await this.saveWavFile(processedOutputPath, this.processedOutputRecording);
    
    const referencePath = path.join(recordingsDir, `reference_${timestamp}.wav`);
    await this.saveWavFile(referencePath, this.referenceRecording);
    
    // Also run AEC3 post-processing for comparison
    let aec3OutputPath: string | undefined;
    try {
      const aec3DemoPath = path.join(process.cwd(), 'vendor', 'aec3', 'build', 'aec3_demo_hq');
      if (fs.existsSync(aec3DemoPath)) {
        // Convert to 16-bit PCM for AEC3
        const rawInput16bit = path.join(recordingsDir, `raw_input_16bit_${timestamp}.wav`);
        const reference16bit = path.join(recordingsDir, `reference_16bit_${timestamp}.wav`);
        aec3OutputPath = path.join(recordingsDir, `aec3_output_${timestamp}.wav`);
        
        execSync(`ffmpeg -y -i "${rawInputPath}" -acodec pcm_s16le "${rawInput16bit}" 2>/dev/null`);
        execSync(`ffmpeg -y -i "${referencePath}" -acodec pcm_s16le "${reference16bit}" 2>/dev/null`);
        
        // Run AEC3 with mode 7 (ultra aggressive)
        execSync(`"${aec3DemoPath}" "${reference16bit}" "${rawInput16bit}" "${aec3OutputPath}" 7`, { timeout: 60000 });
        
        // Cleanup temp files
        try {
          fs.unlinkSync(rawInput16bit);
          fs.unlinkSync(reference16bit);
        } catch (e) {}
        
        console.log(`AEC3 post-processing complete: ${aec3OutputPath}`);
      }
    } catch (e) {
      console.error('AEC3 post-processing failed:', e);
    }
    
    this.rawInputRecording = [];
    this.processedOutputRecording = [];
    this.referenceRecording = [];
    
    return { rawInput: rawInputPath, processedOutput: processedOutputPath, reference: referencePath, aec3Output: aec3OutputPath };
  }

  private async saveWavFile(filePath: string, chunks: Float32Array[]): Promise<void> {
    let totalSamples = 0;
    for (const chunk of chunks) totalSamples += chunk.length;
    
    const header = Buffer.alloc(44);
    const dataSize = totalSamples * 4;
    
    header.write('RIFF', 0);
    header.writeUInt32LE(dataSize + 36, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(3, 20);  // Float format
    header.writeUInt16LE(1, 22);  // Mono
    header.writeUInt32LE(this.config.sampleRate, 24);
    header.writeUInt32LE(this.config.sampleRate * 4, 28);
    header.writeUInt16LE(4, 32);
    header.writeUInt16LE(32, 34);
    header.write('data', 36);
    header.writeUInt32LE(dataSize, 40);
    
    const stream = fs.createWriteStream(filePath);
    stream.write(header);
    
    for (const chunk of chunks) {
      const buffer = Buffer.alloc(chunk.length * 4);
      for (let i = 0; i < chunk.length; i++) {
        buffer.writeFloatLE(chunk[i], i * 4);
      }
      stream.write(buffer);
    }
    
    await new Promise<void>((resolve, reject) => {
      stream.end(() => resolve());
      stream.on('error', reject);
    });
  }

  getIsRecording(): boolean { return this.isRecording; }
}

let processorInstance: SpeexProcessor | null = null;

export function getSpeexProcessor(): SpeexProcessor {
  if (!processorInstance) {
    processorInstance = new SpeexProcessor();
  }
  return processorInstance;
}

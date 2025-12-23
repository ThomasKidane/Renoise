/**
 * WebRTC AEC3-based Audio Processor
 * Uses the compiled AEC3 demo for high-quality echo cancellation
 * Mode 7 = Ultra Aggressive (maximum echo removal)
 */

import * as naudiodon from 'naudiodon2';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { app } from 'electron';
import { DeviceManager, getDeviceManager } from './device-manager';

export interface AEC3ProcessorConfig {
  sampleRate: number;
  frameSize: number;
  aec3Mode: number;  // 0-7, higher = more aggressive
  inputDeviceId: number | null;
  referenceDeviceId: number | null;
  outputDeviceId: number | null;
  inputGain: number;
  referenceGain: number;
  outputGain: number;
  bufferMs: number;  // How much audio to buffer before processing
}

export interface AudioLevels {
  input: number;
  reference: number;
  output: number;
  convergence: number;
}

// Ring buffer for audio data
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

const DEFAULT_CONFIG: AEC3ProcessorConfig = {
  sampleRate: 48000,
  frameSize: 480,     // 10ms at 48kHz
  aec3Mode: 7,        // Ultra aggressive
  inputDeviceId: null,
  referenceDeviceId: null,
  outputDeviceId: null,
  inputGain: 1.0,
  referenceGain: 1.0,
  outputGain: 1.0,
  bufferMs: 500       // 500ms buffer for processing (introduces latency but enables AEC3)
};

export class AEC3Processor extends EventEmitter {
  private config: AEC3ProcessorConfig;
  private deviceManager: DeviceManager;
  
  private inputStream: any = null;
  private referenceStream: any = null;
  private outputStream: any = null;
  
  private inputBuffer: RingBuffer;
  private referenceBuffer: RingBuffer;
  private outputBuffer: RingBuffer;
  
  private isRunning: boolean = false;
  private levels: AudioLevels = { input: 0, reference: 0, output: 0, convergence: 0 };
  private levelLogCounter: number = 0;
  
  private isRecording: boolean = false;
  private rawInputRecording: Float32Array[] = [];
  private processedOutputRecording: Float32Array[] = [];
  private referenceRecording: Float32Array[] = [];
  
  private processTimer: NodeJS.Timeout | null = null;
  private aec3DemoPath: string;
  private tempDir: string;

  constructor(config: Partial<AEC3ProcessorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.deviceManager = getDeviceManager();
    
    // Find AEC3 demo executable
    const appPath = app.getAppPath();
    this.aec3DemoPath = path.join(appPath, 'vendor', 'aec3', 'build', 'aec3_demo_hq');
    
    // Fallback paths
    if (!fs.existsSync(this.aec3DemoPath)) {
      this.aec3DemoPath = path.join(process.cwd(), 'vendor', 'aec3', 'build', 'aec3_demo_hq');
    }
    
    console.log(`AEC3 demo path: ${this.aec3DemoPath}`);
    console.log(`AEC3 exists: ${fs.existsSync(this.aec3DemoPath)}`);
    
    // Create temp directory for processing
    this.tempDir = path.join(app.getPath('temp'), 'aec3-processor');
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
    
    // Buffer size based on config
    const bufferSamples = Math.floor(this.config.sampleRate * (this.config.bufferMs / 1000));
    this.inputBuffer = new RingBuffer(bufferSamples * 2);
    this.referenceBuffer = new RingBuffer(bufferSamples * 2);
    this.outputBuffer = new RingBuffer(bufferSamples * 2);
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    console.log('Starting AEC3Processor...');
    console.log(`Sample rate: ${this.config.sampleRate}Hz, Mode: ${this.config.aec3Mode}`);
    
    if (!fs.existsSync(this.aec3DemoPath)) {
      throw new Error(`AEC3 demo not found at: ${this.aec3DemoPath}`);
    }
    
    try {
      await this.startStreams();
      this.isRunning = true;
      
      // Start processing timer - process chunks periodically
      const processIntervalMs = this.config.bufferMs;
      this.processTimer = setInterval(() => this.processChunk(), processIntervalMs);
      
      console.log('AEC3Processor started successfully');
    } catch (error) {
      console.error('Failed to start AEC3Processor:', error);
      this.stop();
      throw error;
    }
  }

  stop(): void {
    console.log('Stopping AEC3Processor...');
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
    
    this.inputBuffer.clear();
    this.referenceBuffer.clear();
    this.outputBuffer.clear();
    
    console.log('AEC3Processor stopped');
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

    // Input stream (microphone)
    this.inputStream = new naudiodon.AudioIO({
      inOptions: {
        channelCount: 1,
        sampleFormat: naudiodon.SampleFormatFloat32,
        sampleRate: this.config.sampleRate,
        deviceId: inputDevice.id,
        closeOnError: false,
        highwaterMark: this.config.frameSize * 4 * 8
      }
    });

    this.inputStream.on('data', (buffer: Buffer) => this.handleInputData(buffer));
    this.inputStream.on('error', (err: Error) => {
      console.error('Input stream error:', err);
      this.emit('error', { type: 'input', error: err });
    });
    this.inputStream.start();

    // Reference stream (system audio)
    if (referenceDevice) {
      this.referenceStream = new naudiodon.AudioIO({
        inOptions: {
          channelCount: 1,
          sampleFormat: naudiodon.SampleFormatFloat32,
          sampleRate: this.config.sampleRate,
          deviceId: referenceDevice.id,
          closeOnError: false,
          highwaterMark: this.config.frameSize * 4 * 8
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
          highwaterMark: this.config.frameSize * 4 * 8
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

  private processChunk(): void {
    if (!this.isRunning) return;
    
    const chunkSamples = Math.floor(this.config.sampleRate * (this.config.bufferMs / 1000));
    
    // Only process if we have enough samples
    if (this.inputBuffer.getAvailable() < chunkSamples) {
      return;
    }
    
    // Read chunks from buffers
    const inputChunk = this.inputBuffer.read(chunkSamples);
    const referenceChunk = this.referenceBuffer.read(chunkSamples);
    
    const inputRMS = this.calculateRMS(inputChunk);
    const refRMS = this.calculateRMS(referenceChunk);
    
    // Process through AEC3
    let outputChunk: Float32Array;
    
    if (refRMS > 0.005) {
      // Reference audio present - use AEC3
      try {
        outputChunk = this.processWithAEC3(inputChunk, referenceChunk);
      } catch (e) {
        console.error('AEC3 processing failed, using passthrough:', e);
        outputChunk = new Float32Array(inputChunk);
      }
    } else {
      // No reference audio - pass through
      outputChunk = new Float32Array(inputChunk);
    }
    
    // Apply output gain
    for (let i = 0; i < outputChunk.length; i++) {
      outputChunk[i] *= this.config.outputGain;
    }
    
    const outputRMS = this.calculateRMS(outputChunk);
    
    this.levels = {
      input: inputRMS,
      reference: refRMS,
      output: outputRMS,
      convergence: refRMS > 0.001 ? Math.min(1, outputRMS / refRMS) : 1
    };
    
    this.levelLogCounter++;
    if (this.levelLogCounter >= 2) {  // Log every second at 500ms intervals
      console.log(`AEC3 Levels - Input: ${(inputRMS * 100).toFixed(2)}%, Ref: ${(refRMS * 100).toFixed(2)}%, Out: ${(outputRMS * 100).toFixed(2)}%`);
      this.levelLogCounter = 0;
    }
    
    this.emit('levels', this.levels);
    
    // Recording
    if (this.isRecording) {
      this.rawInputRecording.push(new Float32Array(inputChunk));
      this.processedOutputRecording.push(new Float32Array(outputChunk));
      this.referenceRecording.push(new Float32Array(referenceChunk));
    }
    
    // Write to output
    if (this.outputStream) {
      const outputBuffer = Buffer.alloc(outputChunk.length * 4);
      for (let i = 0; i < outputChunk.length; i++) {
        outputBuffer.writeFloatLE(outputChunk[i], i * 4);
      }
      this.outputStream.write(outputBuffer);
    }
  }

  private processWithAEC3(input: Float32Array, reference: Float32Array): Float32Array {
    const timestamp = Date.now();
    const inputPath = path.join(this.tempDir, `input_${timestamp}.wav`);
    const refPath = path.join(this.tempDir, `ref_${timestamp}.wav`);
    const outputPath = path.join(this.tempDir, `output_${timestamp}.wav`);
    
    try {
      // Write input files as 16-bit PCM (required by AEC3)
      this.writeWav16bit(inputPath, input);
      this.writeWav16bit(refPath, reference);
      
      // Run AEC3
      execSync(`"${this.aec3DemoPath}" "${refPath}" "${inputPath}" "${outputPath}" ${this.config.aec3Mode}`, {
        timeout: 5000,
        stdio: 'pipe'
      });
      
      // Read output
      if (fs.existsSync(outputPath)) {
        const result = this.readWav16bit(outputPath);
        
        // Cleanup
        try {
          fs.unlinkSync(inputPath);
          fs.unlinkSync(refPath);
          fs.unlinkSync(outputPath);
          // Cleanup linear.wav that AEC3 creates
          const linearPath = path.join(this.tempDir, 'linear.wav');
          if (fs.existsSync(linearPath)) fs.unlinkSync(linearPath);
        } catch (e) {}
        
        return result;
      }
    } catch (e) {
      console.error('AEC3 processing error:', e);
      // Cleanup on error
      try {
        if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
        if (fs.existsSync(refPath)) fs.unlinkSync(refPath);
        if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath);
      } catch (e2) {}
    }
    
    // Return original on error
    return new Float32Array(input);
  }

  private writeWav16bit(filePath: string, samples: Float32Array): void {
    const header = Buffer.alloc(44);
    const dataSize = samples.length * 2;  // 16-bit = 2 bytes per sample
    
    header.write('RIFF', 0);
    header.writeUInt32LE(dataSize + 36, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);  // PCM format
    header.writeUInt16LE(1, 22);  // Mono
    header.writeUInt32LE(this.config.sampleRate, 24);
    header.writeUInt32LE(this.config.sampleRate * 2, 28);  // Byte rate
    header.writeUInt16LE(2, 32);  // Block align
    header.writeUInt16LE(16, 34); // Bits per sample
    header.write('data', 36);
    header.writeUInt32LE(dataSize, 40);
    
    const data = Buffer.alloc(dataSize);
    for (let i = 0; i < samples.length; i++) {
      // Convert float [-1, 1] to 16-bit int
      const int16 = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
      data.writeInt16LE(int16, i * 2);
    }
    
    fs.writeFileSync(filePath, Buffer.concat([header, data]));
  }

  private readWav16bit(filePath: string): Float32Array {
    const buffer = fs.readFileSync(filePath);
    
    // Skip 44-byte header
    const dataStart = 44;
    const dataSize = buffer.length - dataStart;
    const numSamples = dataSize / 2;
    
    const samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      const int16 = buffer.readInt16LE(dataStart + i * 2);
      samples[i] = int16 / 32768;  // Convert to float [-1, 1]
    }
    
    return samples;
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
  updateConfig(newConfig: Partial<AEC3ProcessorConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
  getConfig(): AEC3ProcessorConfig { return { ...this.config }; }

  startRecording(): void {
    if (this.isRecording) return;
    this.isRecording = true;
    this.rawInputRecording = [];
    this.processedOutputRecording = [];
    this.referenceRecording = [];
    console.log('Recording started');
  }

  async stopRecording(): Promise<{ rawInput: string; processedOutput: string; reference: string }> {
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
    
    this.rawInputRecording = [];
    this.processedOutputRecording = [];
    this.referenceRecording = [];
    
    return { rawInput: rawInputPath, processedOutput: processedOutputPath, reference: referencePath };
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

let processorInstance: AEC3Processor | null = null;

export function getAEC3Processor(): AEC3Processor {
  if (!processorInstance) {
    processorInstance = new AEC3Processor();
  }
  return processorInstance;
}


/**
 * Audio Processor with Deep Learning Echo Cancellation
 * - Live output: DTLN-AEC for real-time processing
 * - Recordings: Demucs + Multi-resolution speaker separation for high-quality post-processing
 *   (removes instruments AND separates your voice from the singer)
 */

import * as naudiodon from 'naudiodon2';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { execSync, spawn } from 'child_process';
import { DeviceManager, getDeviceManager } from './device-manager';
import { DTLNProcessor, getDTLNProcessor } from './dtln-processor';

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
    
    for (let i = toRead; i < count; i++) {
      result[i] = 0;
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
  sampleRate: 48000,
  frameSize: 480,     // 10ms at 48kHz
  filterLength: 4800,
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
  
  private dtln: DTLNProcessor;
  private dtlnReady: boolean = false;
  
  private inputBuffer: RingBuffer;
  private referenceBuffer: RingBuffer;
  
  private isRunning: boolean = false;
  private levels: AudioLevels = { input: 0, reference: 0, output: 0, convergence: 0 };
  private levelLogCounter: number = 0;
  
  private isRecording: boolean = false;
  private rawInputRecording: Float32Array[] = [];
  private processedOutputRecording: Float32Array[] = [];
  private referenceRecording: Float32Array[] = [];
  
  private processTimer: NodeJS.Timeout | null = null;

  constructor(config: Partial<SpeexProcessorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.deviceManager = getDeviceManager();
    
    this.dtln = getDTLNProcessor();
    
    // Buffer size: 200ms worth of audio
    const bufferSize = Math.floor(this.config.sampleRate * 0.2);
    this.inputBuffer = new RingBuffer(bufferSize);
    this.referenceBuffer = new RingBuffer(bufferSize);
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    console.log('Starting Audio Processor (Recording Mode)...');
    console.log(`Sample rate: ${this.config.sampleRate}Hz, Frame size: ${this.config.frameSize}`);
    
    try {
      // Skip DTLN initialization - just record
      this.dtlnReady = false;
      console.log('DTLN disabled - recording mode only');
      
      await this.startStreams();
      this.isRunning = true;
      
      // Process frames at 5ms intervals
      this.processTimer = setInterval(() => this.processFrame(), 5);
      
      console.log('Audio Processor started successfully (recording mode)');
    } catch (error) {
      console.error('Failed to start processor:', error);
      this.stop();
      throw error;
    }
  }

  stop(): void {
    console.log('Stopping Audio Processor...');
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
    
    this.dtln.reset();
    this.inputBuffer.clear();
    this.referenceBuffer.clear();
    
    console.log('Audio Processor stopped');
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
        highwaterMark: this.config.frameSize * 4 * 4
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
    let framesProcessed = 0;
    const maxFramesPerTick = 50;
    
    while (this.inputBuffer.getAvailable() >= frameSize && framesProcessed < maxFramesPerTick) {
      const inputFrame = this.inputBuffer.read(frameSize);
      const referenceFrame = this.referenceBuffer.read(frameSize);
      
      const inputRMS = this.calculateRMS(inputFrame);
      const refRMS = this.calculateRMS(referenceFrame);
      
      // Just pass through input (no DTLN processing)
      let output: Float32Array = inputFrame;
      
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
      
      // Recording - save raw input for SepFormer post-processing
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
    
    // Log every ~1 second
    this.levelLogCounter += framesProcessed;
    if (this.levelLogCounter >= 100) {
      console.log(`[Recording Mode] Frames: ${framesProcessed} | Buffers: in=${this.inputBuffer.getAvailable()}, ref=${this.referenceBuffer.getAvailable()}`);
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

  async stopRecording(): Promise<{ rawInput: string; processedOutput: string; reference: string; dtlnOutput?: string; sepformerOutput?: string }> {
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
    
    const processedOutputPath = path.join(recordingsDir, `dtln_output_${timestamp}.wav`);
    await this.saveWavFile(processedOutputPath, this.processedOutputRecording);
    
    const referencePath = path.join(recordingsDir, `reference_${timestamp}.wav`);
    await this.saveWavFile(referencePath, this.referenceRecording);
    
    console.log(`Recordings saved:`);
    console.log(`  Raw input: ${rawInputPath}`);
    console.log(`  DTLN output (live): ${processedOutputPath}`);
    console.log(`  Reference: ${referencePath}`);
    
    // Run advanced processing (Demucs + Multi-resolution separation) for high-quality output
    const cascadedOutputPath = path.join(recordingsDir, `cascaded_output_${timestamp}.wav`);
    this.runAdvancedProcessing(rawInputPath, referencePath, cascadedOutputPath);
    
    this.rawInputRecording = [];
    this.processedOutputRecording = [];
    this.referenceRecording = [];
    
    return { 
      rawInput: rawInputPath, 
      processedOutput: processedOutputPath, 
      reference: referencePath,
      dtlnOutput: processedOutputPath,
      cascadedOutput: cascadedOutputPath
    };
  }
  
  /**
   * Run advanced processing in background (Demucs + Multi-resolution speaker separation)
   * This removes instruments AND separates your voice from the singer
   */
  private runAdvancedProcessing(micPath: string, refPath: string, outputPath: string): void {
    const scriptsDir = path.join(process.cwd(), 'scripts');
    const advancedScript = path.join(scriptsDir, 'advanced_process.py');
    
    // Check if script exists
    if (!fs.existsSync(advancedScript)) {
      console.warn('[Advanced] Script not found, skipping post-processing');
      return;
    }
    
    // Use system Python (anaconda) which has torch/demucs installed
    const pythonPaths = [
      '/opt/anaconda3/bin/python',
      'python3',
      'python'
    ];
    
    let pythonPath = 'python3';
    for (const p of pythonPaths) {
      if (p.startsWith('/') && fs.existsSync(p)) {
        pythonPath = p;
        break;
      }
    }
    
    console.log(`[Advanced] Starting Demucs + Multi-resolution processing...`);
    console.log(`[Advanced] Mic: ${micPath}`);
    console.log(`[Advanced] Ref: ${refPath}`);
    console.log(`[Advanced] Output: ${outputPath}`);
    
    const advancedProcess = spawn(pythonPath, [
      advancedScript,
      '--mic', micPath,
      '--ref', refPath,
      '--output', outputPath,
      '--debug'  // Save intermediate tracks for debugging
    ], {
      cwd: scriptsDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true
    });
    
    advancedProcess.stdout?.on('data', (data) => {
      console.log(`[Advanced] ${data.toString().trim()}`);
    });
    
    advancedProcess.stderr?.on('data', (data) => {
      console.error(`[Advanced Error] ${data.toString().trim()}`);
    });
    
    advancedProcess.on('exit', (code) => {
      if (code === 0) {
        console.log(`[Advanced] Processing complete: ${outputPath}`);
        this.emit('cascaded-complete', { output: outputPath });
      } else {
        console.error(`[Advanced] Processing failed with code ${code}`);
        this.emit('cascaded-error', { code });
      }
    });
    
    // Don't wait for the process
    advancedProcess.unref();
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

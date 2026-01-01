/**
 * DTLN-AEC Real-Time Processor
 * 
 * High-quality deep learning-based acoustic echo cancellation.
 * Uses a Python WebSocket server running the DTLN model.
 */

import WebSocket from 'ws';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { app } from 'electron';
import { EventEmitter } from 'events';

interface DTLNConfig {
  serverHost: string;
  serverPort: number;
  inputSampleRate: number;
  targetSampleRate: number;
}

const DEFAULT_CONFIG: DTLNConfig = {
  serverHost: 'localhost',
  serverPort: 8765,
  inputSampleRate: 48000,
  targetSampleRate: 16000, // DTLN expects 16kHz
};

export class DTLNProcessor extends EventEmitter {
  private config: DTLNConfig;
  private ws: WebSocket | null = null;
  private serverProcess: ChildProcess | null = null;
  private isConnected: boolean = false;
  private outputBuffer: Float32Array[] = [];
  private connectionPromise: Promise<void> | null = null;
  
  // Resampling buffers
  private resampleRatio: number;

  constructor(config: Partial<DTLNConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.resampleRatio = this.config.targetSampleRate / this.config.inputSampleRate;
  }

  /**
   * Start the DTLN Python server
   */
  private async startServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Get the path relative to the app
      const appPath = app.getAppPath();
      const dtlnPath = path.join(appPath, 'DTRL-AEC', 'DTLN-aec');
      const venvPython = path.join(dtlnPath, 'venv', 'bin', 'python');
      const serverScript = path.join(dtlnPath, 'realtime_server.py');

      console.log(`[DTLN] Starting server from: ${dtlnPath}`);
      console.log(`[DTLN] Python: ${venvPython}`);
      console.log(`[DTLN] Script: ${serverScript}`);

      this.serverProcess = spawn(venvPython, [
        serverScript,
        '--port', this.config.serverPort.toString()
      ], {
        cwd: dtlnPath,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });

      let started = false;
      let startupOutput = '';

      this.serverProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;
        console.log('[DTLN Server]', output.trim());
        
        if (output.includes('Listening on') && !started) {
          started = true;
          setTimeout(resolve, 500); // Give server time to fully initialize
        }
      });

      this.serverProcess.stderr?.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;
        console.error('[DTLN Server Error]', output.trim());
      });

      this.serverProcess.on('error', (err) => {
        console.error('[DTLN] Failed to start server:', err);
        reject(err);
      });

      this.serverProcess.on('exit', (code) => {
        console.log(`[DTLN] Server exited with code ${code}`);
        if (!started) {
          reject(new Error(`Server failed to start. Output: ${startupOutput}`));
        }
        this.isConnected = false;
        this.ws = null;
      });

      // Timeout if server doesn't start
      setTimeout(() => {
        if (!started) {
          console.error('[DTLN] Server startup timeout. Output:', startupOutput);
          this.serverProcess?.kill();
          reject(new Error('DTLN server startup timeout'));
        }
      }, 30000);
    });
  }

  /**
   * Connect to the DTLN WebSocket server
   */
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `ws://${this.config.serverHost}:${this.config.serverPort}`;
      console.log(`[DTLN] Connecting to ${url}`);

      this.ws = new WebSocket(url);

      this.ws.on('open', () => {
        console.log('[DTLN] WebSocket connected');
        this.isConnected = true;
        resolve();
      });

      this.ws.on('message', (data) => {
        try {
          const response = JSON.parse(data.toString());
          if (response.type === 'audio' && response.data) {
            const outputBytes = Buffer.from(response.data, 'base64');
            const output = new Float32Array(
              outputBytes.buffer, 
              outputBytes.byteOffset, 
              outputBytes.length / 4
            );
            this.outputBuffer.push(output);
          }
        } catch (e) {
          console.error('[DTLN] Error parsing response:', e);
        }
      });

      this.ws.on('error', (err) => {
        console.error('[DTLN] WebSocket error:', err);
        if (!this.isConnected) {
          reject(err);
        }
      });

      this.ws.on('close', () => {
        console.log('[DTLN] WebSocket closed');
        this.isConnected = false;
      });

      // Timeout
      setTimeout(() => {
        if (!this.isConnected) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000);
    });
  }

  /**
   * Initialize - connect to server (start if needed)
   */
  async initialize(): Promise<void> {
    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = (async () => {
      try {
        console.log('[DTLN] Initializing...');
        
        // First try to connect to existing server
        try {
          await this.connectWebSocket();
          console.log('[DTLN] Connected to existing server');
          return;
        } catch (e) {
          console.log('[DTLN] No existing server, starting new one...');
        }
        
        // Start server if connection failed
        await this.startServer();
        await this.connectWebSocket();
        console.log('[DTLN] Initialized successfully');
      } catch (e) {
        console.error('[DTLN] Initialization failed:', e);
        this.connectionPromise = null;
        throw e;
      }
    })();

    return this.connectionPromise;
  }

  /**
   * Simple linear resampling
   */
  private resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
    if (fromRate === toRate) return input;
    
    const ratio = toRate / fromRate;
    const outputLength = Math.floor(input.length * ratio);
    const output = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i / ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, input.length - 1);
      const frac = srcIndex - srcIndexFloor;
      
      output[i] = input[srcIndexFloor] * (1 - frac) + input[srcIndexCeil] * frac;
    }
    
    return output;
  }

  /**
   * Process audio through DTLN
   * @param mic Microphone input (with echo)
   * @param ref Reference signal (echo source)
   * @returns Processed audio (clean voice) or null if not ready
   */
  process(mic: Float32Array, ref: Float32Array): Float32Array | null {
    if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      // Pass through if not connected
      return mic;
    }

    try {
      // Resample to 16kHz for DTLN
      const micResampled = this.resample(mic, this.config.inputSampleRate, this.config.targetSampleRate);
      const refResampled = this.resample(ref, this.config.inputSampleRate, this.config.targetSampleRate);

      // Send to server
      const message = JSON.stringify({
        type: 'audio',
        mic: Buffer.from(micResampled.buffer, micResampled.byteOffset, micResampled.byteLength).toString('base64'),
        ref: Buffer.from(refResampled.buffer, refResampled.byteOffset, refResampled.byteLength).toString('base64')
      });

      this.ws.send(message);
    } catch (e) {
      console.error('[DTLN] Error sending audio:', e);
      return mic;
    }

    // Return buffered output (there will be some latency)
    if (this.outputBuffer.length > 0) {
      const output16k = this.outputBuffer.shift()!;
      // Resample back to original rate
      return this.resample(output16k, this.config.targetSampleRate, this.config.inputSampleRate);
    }

    // Return silence while waiting for first output (initial latency)
    return new Float32Array(mic.length);
  }

  /**
   * Reset the DTLN state
   */
  reset(): void {
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'reset' }));
      } catch (e) {
        console.error('[DTLN] Error sending reset:', e);
      }
    }
    this.outputBuffer = [];
  }

  /**
   * Stop the processor and server
   */
  async stop(): Promise<void> {
    console.log('[DTLN] Stopping...');
    
    if (this.ws) {
      try {
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }
    
    if (this.serverProcess) {
      try {
        this.serverProcess.kill('SIGTERM');
        // Give it a moment to clean up
        await new Promise(resolve => setTimeout(resolve, 500));
        if (!this.serverProcess.killed) {
          this.serverProcess.kill('SIGKILL');
        }
      } catch (e) {}
      this.serverProcess = null;
    }
    
    this.isConnected = false;
    this.connectionPromise = null;
    this.outputBuffer = [];
    
    console.log('[DTLN] Stopped');
  }

  /**
   * Check if connected and ready
   */
  isReady(): boolean {
    return this.isConnected && this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get number of buffered output frames
   */
  getBufferedFrames(): number {
    return this.outputBuffer.length;
  }
}

// Singleton instance
let dtlnInstance: DTLNProcessor | null = null;

export function getDTLNProcessor(): DTLNProcessor {
  if (!dtlnInstance) {
    dtlnInstance = new DTLNProcessor();
  }
  return dtlnInstance;
}

export async function initializeDTLN(): Promise<DTLNProcessor> {
  const processor = getDTLNProcessor();
  await processor.initialize();
  return processor;
}

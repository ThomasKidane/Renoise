/**
 * SpeexDSP Echo Cancellation Wrapper
 */

import * as path from 'path';
import { createRequire } from 'module';

let nativeModule: any = null;

function loadNativeModule(): any {
  const nodeRequire = createRequire(import.meta.url || __filename);
  const cwd = process.cwd();
  const possiblePaths = [
    path.join(cwd, 'native', 'build', 'Release', 'speex_aec.node'),
    path.resolve(cwd, 'native/build/Release/speex_aec.node'),
  ];

  for (const modulePath of possiblePaths) {
    try {
      console.log(`Trying to load native module from: ${modulePath}`);
      const mod = nodeRequire(modulePath);
      console.log('Native SpeexAEC module loaded successfully');
      return mod;
    } catch (e: any) {
      console.log(`  -> Failed: ${e.message}`);
    }
  }
  return null;
}

try {
  nativeModule = loadNativeModule();
  if (!nativeModule) {
    console.error('Failed to load native SpeexAEC module');
    console.error('Using fallback passthrough mode');
  }
} catch (error) {
  console.error('Failed to load native SpeexAEC module:', error);
}

export interface SpeexAECOptions {
  frameSize?: number;
  sampleRate?: number;
  filterLength?: number;
}

export class SpeexAEC {
  private native: any = null;
  private frameSize: number;
  private sampleRate: number;

  constructor(options: SpeexAECOptions = {}) {
    this.frameSize = options.frameSize || 160;
    this.sampleRate = options.sampleRate || 16000;

    if (nativeModule) {
      this.native = new nativeModule.SpeexAEC({
        frameSize: this.frameSize,
        sampleRate: this.sampleRate,
        filterLength: options.filterLength || 4096
      });
    }
  }

  process(input: Float32Array, reference: Float32Array): Float32Array {
    if (!this.native) {
      return input;
    }
    return this.native.process(input, reference);
  }

  reset(): void {
    if (this.native) {
      this.native.reset();
    }
  }

  isNativeAvailable(): boolean {
    return this.native !== null;
  }
}

export default SpeexAEC;

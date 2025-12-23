import * as naudiodon from 'naudiodon2';

export interface AudioDevice {
  id: number;
  name: string;
  maxInputChannels: number;
  maxOutputChannels: number;
  defaultSampleRate: number;
  hostAPIName: string;
  isInput: boolean;
  isOutput: boolean;
}

export class DeviceManager {
  private devices: AudioDevice[] = [];

  constructor() {
    this.refreshDevices();
  }

  refreshDevices(): void {
    this.devices = [];
    const rawDevices = naudiodon.getDevices();
    
    for (let i = 0; i < rawDevices.length; i++) {
      const device = rawDevices[i];
      this.devices.push({
        id: device.id,
        name: device.name,
        maxInputChannels: device.maxInputChannels,
        maxOutputChannels: device.maxOutputChannels,
        defaultSampleRate: device.defaultSampleRate,
        hostAPIName: device.hostAPIName,
        isInput: device.maxInputChannels > 0,
        isOutput: device.maxOutputChannels > 0
      });
    }
  }

  getAllDevices(): AudioDevice[] {
    return [...this.devices];
  }

  getInputDevices(): AudioDevice[] {
    return this.devices.filter(d => d.isInput);
  }

  getOutputDevices(): AudioDevice[] {
    return this.devices.filter(d => d.isOutput);
  }

  findDeviceByName(name: string): AudioDevice | undefined {
    return this.devices.find(d => d.name.toLowerCase().includes(name.toLowerCase()));
  }

  getDeviceById(id: number): AudioDevice | undefined {
    return this.devices.find(d => d.id === id);
  }

  logDevices(): void {
    console.log('\n=== Audio Devices ===');
    console.log('\nInput Devices:');
    this.getInputDevices().forEach(d => {
      console.log(`  [${d.id}] ${d.name} (${d.maxInputChannels}ch, ${d.defaultSampleRate}Hz)`);
    });
    console.log('\nOutput Devices:');
    this.getOutputDevices().forEach(d => {
      console.log(`  [${d.id}] ${d.name} (${d.maxOutputChannels}ch, ${d.defaultSampleRate}Hz)`);
    });
  }
}

let deviceManagerInstance: DeviceManager | null = null;

export function getDeviceManager(): DeviceManager {
  if (!deviceManagerInstance) {
    deviceManagerInstance = new DeviceManager();
  }
  return deviceManagerInstance;
}

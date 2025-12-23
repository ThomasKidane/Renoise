import { contextBridge, ipcRenderer } from 'electron';

export interface ElectronAPI {
  getDevices: () => Promise<any>;
  getConfig: () => Promise<any>;
  getLevels: () => Promise<any>;
  getStatus: () => Promise<any>;
  startProcessing: () => Promise<any>;
  stopProcessing: () => Promise<any>;
  setInputDevice: (deviceId: number) => Promise<any>;
  setReferenceDevice: (deviceId: number | null) => Promise<any>;
  setOutputDevice: (deviceId: number | null) => Promise<any>;
  setInputGain: (gain: number) => Promise<any>;
  setReferenceGain: (gain: number) => Promise<any>;
  setStepSize: (mu: number) => Promise<any>;
  setReferenceDelay: (ms: number) => Promise<any>;
  resetFilter: () => Promise<any>;
  startRecording: () => Promise<any>;
  stopRecording: () => Promise<any>;
  getRecordingStatus: () => Promise<any>;
  hideWindow: () => Promise<any>;
  onLevelsUpdate: (callback: (levels: any) => void) => void;
  onProcessingStatus: (callback: (status: any) => void) => void;
  onError: (callback: (error: any) => void) => void;
}

const electronAPI: ElectronAPI = {
  getDevices: () => ipcRenderer.invoke('get-devices'),
  getConfig: () => ipcRenderer.invoke('get-config'),
  getLevels: () => ipcRenderer.invoke('get-levels'),
  getStatus: () => ipcRenderer.invoke('get-status'),
  startProcessing: () => ipcRenderer.invoke('start-processing'),
  stopProcessing: () => ipcRenderer.invoke('stop-processing'),
  setInputDevice: (deviceId) => ipcRenderer.invoke('set-input-device', deviceId),
  setReferenceDevice: (deviceId) => ipcRenderer.invoke('set-reference-device', deviceId),
  setOutputDevice: (deviceId) => ipcRenderer.invoke('set-output-device', deviceId),
  setInputGain: (gain) => ipcRenderer.invoke('set-input-gain', gain),
  setReferenceGain: (gain) => ipcRenderer.invoke('set-reference-gain', gain),
  setStepSize: (mu) => ipcRenderer.invoke('set-step-size', mu),
  setReferenceDelay: (ms) => ipcRenderer.invoke('set-reference-delay', ms),
  resetFilter: () => ipcRenderer.invoke('reset-filter'),
  startRecording: () => ipcRenderer.invoke('start-recording'),
  stopRecording: () => ipcRenderer.invoke('stop-recording'),
  getRecordingStatus: () => ipcRenderer.invoke('get-recording-status'),
  hideWindow: () => ipcRenderer.invoke('hide-window'),
  onLevelsUpdate: (callback) => ipcRenderer.on('levels-update', (_event, levels) => callback(levels)),
  onProcessingStatus: (callback) => ipcRenderer.on('processing-status', (_event, status) => callback(status)),
  onError: (callback) => ipcRenderer.on('error', (_event, error) => callback(error))
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);

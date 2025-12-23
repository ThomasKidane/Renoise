import React, { useState, useEffect, useCallback } from 'react';

interface AudioDevice {
  id: number;
  name: string;
  maxInputChannels: number;
  maxOutputChannels: number;
}

interface AudioLevels {
  input: number;
  reference: number;
  output: number;
  convergence: number;
}

interface ElectronAPI {
  getDevices: () => Promise<{ input: AudioDevice[]; output: AudioDevice[]; all: AudioDevice[] }>;
  getStatus: () => Promise<{ isRunning: boolean }>;
  startProcessing: () => Promise<any>;
  stopProcessing: () => Promise<any>;
  setInputDevice: (id: number) => Promise<any>;
  setReferenceDevice: (id: number | null) => Promise<any>;
  setOutputDevice: (id: number | null) => Promise<any>;
  startRecording: () => Promise<any>;
  stopRecording: () => Promise<any>;
  getRecordingStatus: () => Promise<{ isRecording: boolean }>;
  onLevelsUpdate: (callback: (levels: AudioLevels) => void) => void;
  onProcessingStatus: (callback: (status: { isRunning: boolean }) => void) => void;
  onError: (callback: (error: { message: string }) => void) => void;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

const App: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [levels, setLevels] = useState<AudioLevels>({ input: 0, reference: 0, output: 0, convergence: 0 });
  const [devices, setDevices] = useState<{ input: AudioDevice[]; output: AudioDevice[] }>({ input: [], output: [] });
  const [selectedInput, setSelectedInput] = useState<number | null>(null);
  const [selectedReference, setSelectedReference] = useState<number | null>(null);
  const [selectedOutput, setSelectedOutput] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const api = window.electronAPI;
    if (!api) return;

    api.getDevices().then((devs) => {
      setDevices({ input: devs.input, output: devs.output });
    });

    api.getStatus().then((status) => setIsRunning(status.isRunning));
    api.getRecordingStatus().then((status) => setIsRecording(status.isRecording));

    api.onLevelsUpdate((lvls) => setLevels(lvls));
    api.onProcessingStatus((status) => setIsRunning(status.isRunning));
    api.onError((err) => {
      setError(err.message);
      setTimeout(() => setError(null), 5000);
    });
  }, []);

  const handleStart = useCallback(async () => {
    await window.electronAPI?.startProcessing();
  }, []);

  const handleStop = useCallback(async () => {
    await window.electronAPI?.stopProcessing();
  }, []);

  const handleInputChange = useCallback(async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = parseInt(e.target.value);
    setSelectedInput(id);
    await window.electronAPI?.setInputDevice(id);
  }, []);

  const handleReferenceChange = useCallback(async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value ? parseInt(e.target.value) : null;
    setSelectedReference(id);
    await window.electronAPI?.setReferenceDevice(id);
  }, []);

  const handleOutputChange = useCallback(async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value ? parseInt(e.target.value) : null;
    setSelectedOutput(id);
    await window.electronAPI?.setOutputDevice(id);
  }, []);

  const handleRecord = useCallback(async () => {
    if (isRecording) {
      await window.electronAPI?.stopRecording();
      setIsRecording(false);
    } else {
      await window.electronAPI?.startRecording();
      setIsRecording(true);
    }
  }, [isRecording]);

  return (
    <div className="app">
      <div className="header">
        <h1 className="title">Wispr Audio Separator</h1>
        <span className={`status-badge ${isRunning ? 'active' : 'inactive'}`}>
          {isRunning ? 'Active' : 'Inactive'}
        </span>
      </div>

      {error && (
        <div style={{ background: 'rgba(255,68,68,0.2)', padding: '10px', borderRadius: '6px', marginBottom: '15px', fontSize: '12px', color: '#ff6b6b' }}>
          {error}
        </div>
      )}

      <div className="section">
        <h2 className="section-title">Audio Levels</h2>
        <div className="levels-container">
          <div className="level-meter">
            <div className="level-label">Input</div>
            <div className="level-bar-container">
              <div className="level-bar input" style={{ width: `${Math.min(100, levels.input * 100)}%` }} />
            </div>
            <div className="level-value">{(levels.input * 100).toFixed(1)}%</div>
          </div>
          <div className="level-meter">
            <div className="level-label">Reference</div>
            <div className="level-bar-container">
              <div className="level-bar reference" style={{ width: `${Math.min(100, levels.reference * 100)}%` }} />
            </div>
            <div className="level-value">{(levels.reference * 100).toFixed(1)}%</div>
          </div>
          <div className="level-meter">
            <div className="level-label">Output</div>
            <div className="level-bar-container">
              <div className="level-bar output" style={{ width: `${Math.min(100, levels.output * 100)}%` }} />
            </div>
            <div className="level-value">{(levels.output * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      <div className="section">
        <h2 className="section-title">Devices</h2>
        <div className="device-selector">
          <label>Microphone Input</label>
          <select value={selectedInput ?? ''} onChange={handleInputChange}>
            <option value="">Select input...</option>
            {devices.input.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </div>
        <div className="device-selector">
          <label>Reference (System Audio via BlackHole 2ch)</label>
          <select value={selectedReference ?? ''} onChange={handleReferenceChange}>
            <option value="">Select reference...</option>
            {devices.input.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </div>
        <div className="device-selector">
          <label>Output (to BlackHole 16ch)</label>
          <select value={selectedOutput ?? ''} onChange={handleOutputChange}>
            <option value="">Select output...</option>
            {devices.output.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="controls">
        {isRunning ? (
          <button className="btn btn-danger" onClick={handleStop}>Stop</button>
        ) : (
          <button className="btn btn-primary" onClick={handleStart}>Start</button>
        )}
        <button className={`btn btn-record ${isRecording ? 'recording' : ''}`} onClick={handleRecord}>
          {isRecording ? 'Stop Recording' : 'Record'}
        </button>
      </div>
    </div>
  );
};

export default App;

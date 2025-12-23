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

const PlayIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
);

const StopIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

const RecordIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor">
    <circle cx="12" cy="12" r="8" />
  </svg>
);

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
        <h1 className="title">Renoise</h1>
        <p className="subtitle">Audio source separation</p>
      </div>

      <div className="main-content">
        {error && (
          <div className="error-banner">{error}</div>
        )}

        <div className="status">
          <div className={`status-dot ${isRunning ? 'active' : ''}`} />
          <span className="status-text">{isRunning ? 'Processing' : 'Idle'}</span>
        </div>

        <div className="controls">
          {isRunning ? (
            <button className="btn btn-danger" onClick={handleStop}>
              <StopIcon /> Stop
            </button>
          ) : (
            <button className="btn btn-primary" onClick={handleStart}>
              <PlayIcon /> Start
            </button>
          )}
          <button 
            className={`btn btn-record ${isRecording ? 'recording' : ''}`} 
            onClick={handleRecord}
          >
            <RecordIcon /> {isRecording ? 'Stop' : 'Record'}
          </button>
        </div>

        <div className="devices">
          <div className="device-row">
            <label className="device-label">Microphone Input</label>
            <select 
              className="device-select" 
              value={selectedInput ?? ''} 
              onChange={handleInputChange}
            >
              <option value="">Select input device...</option>
              {devices.input.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
          <div className="device-row">
            <label className="device-label">Reference (System Audio)</label>
            <select 
              className="device-select" 
              value={selectedReference ?? ''} 
              onChange={handleReferenceChange}
            >
              <option value="">Select reference device...</option>
              {devices.input.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
          <div className="device-row">
            <label className="device-label">Output Destination</label>
            <select 
              className="device-select" 
              value={selectedOutput ?? ''} 
              onChange={handleOutputChange}
            >
              <option value="">Select output device...</option>
              {devices.output.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="levels">
          <div className="level-card">
            <div className="level-header">
              <span className="level-label">Input</span>
              <span className="level-value">{(levels.input * 100).toFixed(0)}%</span>
            </div>
            <div className="level-bar-bg">
              <div 
                className="level-bar-fill input" 
                style={{ width: `${Math.min(100, levels.input * 100)}%` }} 
              />
            </div>
          </div>
          <div className="level-card">
            <div className="level-header">
              <span className="level-label">Reference</span>
              <span className="level-value">{(levels.reference * 100).toFixed(0)}%</span>
            </div>
            <div className="level-bar-bg">
              <div 
                className="level-bar-fill reference" 
                style={{ width: `${Math.min(100, levels.reference * 100)}%` }} 
              />
            </div>
          </div>
          <div className="level-card">
            <div className="level-header">
              <span className="level-label">Output</span>
              <span className="level-value">{(levels.output * 100).toFixed(0)}%</span>
            </div>
            <div className="level-bar-bg">
              <div 
                className="level-bar-fill output" 
                style={{ width: `${Math.min(100, levels.output * 100)}%` }} 
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;

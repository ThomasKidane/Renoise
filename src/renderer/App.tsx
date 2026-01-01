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

interface Recording {
  id: string;
  timestamp: string;
  date: string;
  rawInput: string;
  processed: string;
  reference: string;
  demucs?: string;
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
  getRecordings: () => Promise<Recording[]>;
  deleteRecording: (id: string) => Promise<any>;
  getAudioFileUrl: (filePath: string) => Promise<string>;
  onLevelsUpdate: (callback: (levels: AudioLevels) => void) => void;
  onProcessingStatus: (callback: (status: { isRunning: boolean }) => void) => void;
  onError: (callback: (error: { message: string }) => void) => void;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

type NavItem = 'home' | 'recordings' | 'devices';

const App: React.FC = () => {
  const [activeNav, setActiveNav] = useState<NavItem>('home');
  const [isRunning, setIsRunning] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [levels, setLevels] = useState<AudioLevels>({ input: 0, reference: 0, output: 0, convergence: 0 });
  const [devices, setDevices] = useState<{ input: AudioDevice[]; output: AudioDevice[] }>({ input: [], output: [] });
  const [selectedInput, setSelectedInput] = useState<number | null>(null);
  const [selectedReference, setSelectedReference] = useState<number | null>(null);
  const [selectedOutput, setSelectedOutput] = useState<number | null>(null);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null);
  const [audioUrls, setAudioUrls] = useState<{ raw: string; processed: string; demucs?: string } | null>(null);
  const [audioDurations, setAudioDurations] = useState<{ raw: number; processed: number; demucs: number }>({ raw: 0, processed: 0, demucs: 0 });
  const [error, setError] = useState<string | null>(null);

  const formatDuration = (seconds: number): string => {
    if (!seconds || isNaN(seconds) || !isFinite(seconds)) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAudioLoadedMetadata = (type: 'raw' | 'processed' | 'demucs') => (e: React.SyntheticEvent<HTMLAudioElement>) => {
    const audio = e.currentTarget;
    setAudioDurations(prev => ({ ...prev, [type]: audio.duration }));
  };

  const loadRecordings = useCallback(async () => {
    const recs = await window.electronAPI?.getRecordings();
    if (recs) setRecordings(recs);
  }, []);

  // Load audio URLs when a recording is selected
  useEffect(() => {
    const loadAudioUrls = async () => {
      if (!selectedRecording || !window.electronAPI) {
        setAudioUrls(null);
        setAudioDurations({ raw: 0, processed: 0, demucs: 0 });
        return;
      }
      
      // Reset durations when loading new recording
      setAudioDurations({ raw: 0, processed: 0, demucs: 0 });
      
      const [rawUrl, processedUrl] = await Promise.all([
        window.electronAPI.getAudioFileUrl(selectedRecording.rawInput),
        window.electronAPI.getAudioFileUrl(selectedRecording.processed)
      ]);
      
      let demucsUrl: string | undefined;
      if (selectedRecording.demucs) {
        demucsUrl = await window.electronAPI.getAudioFileUrl(selectedRecording.demucs);
      }
      
      setAudioUrls({ raw: rawUrl, processed: processedUrl, demucs: demucsUrl });
    };
    
    loadAudioUrls();
  }, [selectedRecording]);

  useEffect(() => {
    const api = window.electronAPI;
    if (!api) return;

    api.getDevices().then((devs) => {
      setDevices({ input: devs.input, output: devs.output });
    });

    api.getStatus().then((status) => setIsRunning(status.isRunning));
    api.getRecordingStatus().then((status) => setIsRecording(status.isRecording));
    loadRecordings();

    api.onLevelsUpdate((lvls) => setLevels(lvls));
    api.onProcessingStatus((status) => setIsRunning(status.isRunning));
    api.onError((err) => {
      setError(err.message);
      setTimeout(() => setError(null), 5000);
    });
  }, [loadRecordings]);

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
      setTimeout(loadRecordings, 500);
    } else {
      await window.electronAPI?.startRecording();
      setIsRecording(true);
    }
  }, [isRecording, loadRecordings]);

  const handleDeleteRecording = useCallback(async (id: string) => {
    await window.electronAPI?.deleteRecording(id);
    if (selectedRecording?.id === id) {
      setSelectedRecording(null);
    }
    loadRecordings();
  }, [selectedRecording, loadRecordings]);

  const formatDate = (dateStr: string) => {
    try {
      const parts = dateStr.split(' ');
      const datePart = parts[0].replace(/:/g, '-');
      const timePart = parts[1];
      return `${datePart} at ${timePart}`;
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
            <span>Renoise</span>
          </div>
        </div>

        <div className="nav-items">
          <button 
            className={`nav-item ${activeNav === 'home' ? 'active' : ''}`}
            onClick={() => setActiveNav('home')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            <span>Home</span>
          </button>

          <button 
            className={`nav-item ${activeNav === 'recordings' ? 'active' : ''}`}
            onClick={() => { setActiveNav('recordings'); loadRecordings(); }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
            <span>Recordings</span>
          </button>

          <button 
            className={`nav-item ${activeNav === 'devices' ? 'active' : ''}`}
            onClick={() => setActiveNav('devices')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span>Devices</span>
          </button>
        </div>

        <div className="sidebar-footer">
          <div className={`status-badge ${isRunning ? 'active' : ''}`}>
            <div className="status-dot" />
            <span>{isRunning ? 'Processing' : 'Idle'}</span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main">
        {error && (
          <div className="error-banner">{error}</div>
        )}

        {activeNav === 'home' && (
          <div className="page">
            <div className="page-header">
              <h1>Welcome back</h1>
              <div className="stats">
                <div className="stat">
                  <span className="stat-icon">🎙</span>
                  <span className="stat-value">{recordings.length}</span>
                  <span className="stat-label">recordings</span>
                </div>
              </div>
            </div>

            <div className="tip-card">
              <h2>Hold <span className="key">⌘</span> <span className="key">Shift</span> → to start processing</h2>
              <p>
                Renoise uses <strong>DTLN-AEC</strong> for real-time echo cancellation, and <strong>Demucs</strong> (Meta's 
                state-of-the-art model) for high-quality offline vocal extraction on your recordings.
              </p>
              <button className="btn btn-dark" onClick={isRunning ? handleStop : handleStart}>
                {isRunning ? 'Stop Processing' : 'Start Processing'}
              </button>
            </div>

            <div className="controls-row">
              <button 
                className={`btn btn-large ${isRunning ? 'btn-danger' : 'btn-primary'}`}
                onClick={isRunning ? handleStop : handleStart}
              >
                {isRunning ? (
                  <>
                    <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
                    Stop
                  </>
                ) : (
                  <>
                    <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                    Start
                  </>
                )}
              </button>

              <button 
                className={`btn btn-large ${isRecording ? 'btn-recording' : 'btn-outline'}`}
                onClick={handleRecord}
              >
                <svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="8" /></svg>
                {isRecording ? 'Stop Recording' : 'Record'}
              </button>
            </div>

            <div className="levels-section">
              <h3>Audio Levels</h3>
              <div className="levels-grid">
                <div className="level-item">
                  <div className="level-info">
                    <span className="level-name">Input</span>
                    <span className="level-value">{(levels.input * 100).toFixed(0)}%</span>
                  </div>
                  <div className="level-bar">
                    <div className="level-fill input" style={{ width: `${Math.min(100, levels.input * 100)}%` }} />
                  </div>
                </div>
                <div className="level-item">
                  <div className="level-info">
                    <span className="level-name">Reference</span>
                    <span className="level-value">{(levels.reference * 100).toFixed(0)}%</span>
                  </div>
                  <div className="level-bar">
                    <div className="level-fill reference" style={{ width: `${Math.min(100, levels.reference * 100)}%` }} />
                  </div>
                </div>
                <div className="level-item">
                  <div className="level-info">
                    <span className="level-name">Output</span>
                    <span className="level-value">{(levels.output * 100).toFixed(0)}%</span>
                  </div>
                  <div className="level-bar">
                    <div className="level-fill output" style={{ width: `${Math.min(100, levels.output * 100)}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeNav === 'recordings' && (
          <div className="page recordings-page">
            <div className="page-header">
              <h1>Recordings</h1>
              <p className="page-subtitle">Compare original and processed audio side by side</p>
            </div>

            <div className="recordings-layout">
              <div className="recordings-list">
                {recordings.length === 0 ? (
                  <div className="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M19 11a7 7 0 0 1-7 7m0 0a7 7 0 0 1-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 0 1-3-3V5a3 3 0 0 1 6 0v6a3 3 0 0 1-3 3z" />
                    </svg>
                    <p>No recordings yet</p>
                    <span>Start recording to see your files here</span>
                  </div>
                ) : (
                  recordings.map((rec) => (
                    <div 
                      key={rec.id} 
                      className={`recording-item ${selectedRecording?.id === rec.id ? 'selected' : ''}`}
                      onClick={() => setSelectedRecording(rec)}
                    >
                      <div className="recording-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 18V5l12-2v13" />
                          <circle cx="6" cy="18" r="3" />
                          <circle cx="18" cy="16" r="3" />
                        </svg>
                      </div>
                      <div className="recording-info">
                        <span className="recording-date">{formatDate(rec.date)}</span>
                        <span className="recording-id">ID: {rec.id.slice(0, 16)}...</span>
                      </div>
                      <button 
                        className="delete-btn"
                        onClick={(e) => { e.stopPropagation(); handleDeleteRecording(rec.id); }}
                      >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                      </button>
                    </div>
                  ))
                )}
              </div>

              <div className="recording-preview">
                {selectedRecording ? (
                  <>
                    <div className="preview-header">
                      <h3>Compare Audio</h3>
                      <span className="preview-date">{formatDate(selectedRecording.date)}</span>
                    </div>
                    
                    <div className="audio-compare">
                      <div className="audio-track">
                        <div className="track-header">
                          <span className="track-badge original">A</span>
                          <span className="track-label">Original (Raw Input)</span>
                          <span className="track-duration">{formatDuration(audioDurations.raw)}</span>
                        </div>
                        <audio 
                          controls 
                          src={audioUrls?.raw || ''} 
                          key={`raw-${selectedRecording.id}`}
                          onLoadedMetadata={handleAudioLoadedMetadata('raw')}
                        />
                      </div>
                      
                      <div className="audio-track">
                        <div className="track-header">
                          <span className="track-badge processed">B</span>
                          <span className="track-label">DTLN (Live Processing)</span>
                          <span className="track-duration">{formatDuration(audioDurations.processed)}</span>
                        </div>
                        <audio 
                          controls 
                          src={audioUrls?.processed || ''} 
                          key={`processed-${selectedRecording.id}`}
                          onLoadedMetadata={handleAudioLoadedMetadata('processed')}
                        />
                      </div>
                      
                      {audioUrls?.demucs ? (
                        <div className="audio-track demucs">
                          <div className="track-header">
                            <span className="track-badge demucs">C</span>
                            <span className="track-label">Demucs (High Quality)</span>
                            <span className="track-duration">{formatDuration(audioDurations.demucs)}</span>
                          </div>
                          <audio 
                            controls 
                            src={audioUrls.demucs} 
                            key={`demucs-${selectedRecording.id}`}
                            onLoadedMetadata={handleAudioLoadedMetadata('demucs')}
                          />
                        </div>
                      ) : selectedRecording.demucs === undefined ? (
                        <div className="audio-track processing">
                          <div className="track-header">
                            <span className="track-badge demucs">C</span>
                            <span className="track-label">Demucs (Processing...)</span>
                          </div>
                          <div className="processing-indicator">
                            <div className="spinner" />
                            <span>High-quality vocal extraction in progress...</span>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <div className="preview-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />
                      <path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    <p>Select a recording to preview</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeNav === 'devices' && (
          <div className="page">
            <div className="page-header">
              <h1>Device Settings</h1>
              <p className="page-subtitle">Configure your audio input and output devices</p>
            </div>

            <div className="settings-grid">
              <div className="setting-card">
                <div className="setting-icon mic">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                  </svg>
                </div>
                <div className="setting-content">
                  <label className="setting-label">Microphone Input</label>
                  <p className="setting-desc">Your voice input device</p>
                  <select 
                    className="setting-select" 
                    value={selectedInput ?? ''} 
                    onChange={handleInputChange}
                  >
                    <option value="">Select input device...</option>
                    {devices.input.map((d) => (
                      <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="setting-card">
                <div className="setting-icon ref">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
                  </svg>
                </div>
                <div className="setting-content">
                  <label className="setting-label">Reference (System Audio)</label>
                  <p className="setting-desc">Audio to remove from your voice</p>
                  <select 
                    className="setting-select" 
                    value={selectedReference ?? ''} 
                    onChange={handleReferenceChange}
                  >
                    <option value="">Select reference device...</option>
                    {devices.input.map((d) => (
                      <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="setting-card">
                <div className="setting-icon out">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 18v-6a9 9 0 0 1 18 0v6" />
                    <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z" />
                  </svg>
                </div>
                <div className="setting-content">
                  <label className="setting-label">Output Destination</label>
                  <p className="setting-desc">Where to send processed audio</p>
                  <select 
                    className="setting-select" 
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
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;

import { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, systemPreferences } from 'electron';
import * as path from 'path';
import { getSpeexProcessor, SpeexProcessor } from './speex-processor';
import { getDeviceManager, DeviceManager } from './device-manager';

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let audioProcessor: SpeexProcessor;
let deviceManager: DeviceManager;

async function checkMicrophonePermission(): Promise<boolean> {
  if (process.platform !== 'darwin') return true;

  const status = systemPreferences.getMediaAccessStatus('microphone');
  console.log(`Microphone permission status: ${status}`);

  if (status === 'granted') return true;

  if (status === 'not-determined') {
    console.log('Requesting microphone permission...');
    const granted = await systemPreferences.askForMediaAccess('microphone');
    console.log(`Microphone permission ${granted ? 'granted' : 'denied'}`);
    return granted;
  }

  console.error('Microphone permission denied');
  return false;
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 380,
    height: 520,
    show: false,
    frame: true,
    resizable: true,
    skipTaskbar: false,
    alwaysOnTop: false,
    backgroundColor: '#0a0a0f',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  // Window stays open - user must click tray icon to toggle visibility
}

function createTray(): void {
  const icon = nativeImage.createFromBuffer(
    Buffer.from([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0xF3, 0xFF, 0x61, 0x00, 0x00, 0x00,
      0x01, 0x73, 0x52, 0x47, 0x42, 0x00, 0xAE, 0xCE, 0x1C, 0xE9, 0x00, 0x00,
      0x00, 0x6B, 0x49, 0x44, 0x41, 0x54, 0x38, 0x8D, 0xED, 0xD2, 0xB1, 0x0D,
      0xC0, 0x20, 0x0C, 0x04, 0xD0, 0x4F, 0x91, 0x82, 0x8A, 0x25, 0xB2, 0x42,
      0x46, 0x60, 0x84, 0x8C, 0x90, 0x15, 0x52, 0x50, 0x20, 0x0A, 0x22, 0x04,
      0x45, 0x12, 0x45, 0x9A, 0x14, 0x79, 0x9D, 0x25, 0xFB, 0x2C, 0x1B, 0x00,
      0xFC, 0x0B, 0x8D, 0x88, 0xD8, 0x91, 0x74, 0x26, 0xDD, 0x24, 0x1D, 0xAC,
      0x55, 0x66, 0x76, 0xB1, 0x56, 0x99, 0xD9, 0xC5, 0x5A, 0x65, 0x66, 0x17,
      0x6B, 0x95, 0x99, 0x5D, 0xAC, 0x55, 0x66, 0x76, 0xB1, 0x56, 0x99, 0xD9,
      0xC5, 0x5A, 0x65, 0x66, 0x17, 0x6B, 0x95, 0x99, 0x5D, 0xAC, 0x55, 0x66,
      0x76, 0xB1, 0x56, 0x99, 0xD9, 0x45, 0x44, 0x6C, 0x00, 0xBE, 0x03, 0x0F,
      0x60, 0x06, 0x01, 0x5E, 0xA0, 0x1C, 0x32, 0x00, 0x00, 0x00, 0x00, 0x49,
      0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ])
  );

  tray = new Tray(icon);
  tray.setToolTip('Wispr Audio Separator');

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show Window', click: () => toggleWindow() },
    { type: 'separator' },
    { label: 'Start Processing', click: () => startProcessing() },
    { label: 'Stop Processing', click: () => stopProcessing() },
    { type: 'separator' },
    { label: 'Quit', click: () => { audioProcessor?.stop(); app.quit(); } }
  ]);

  tray.setContextMenu(contextMenu);
  tray.on('click', () => toggleWindow());
}

function toggleWindow(): void {
  if (!mainWindow) {
    createWindow();
    return;
  }
  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    showWindow();
  }
}

function showWindow(): void {
  if (!mainWindow) return;
  const trayBounds = tray?.getBounds();
  const windowBounds = mainWindow.getBounds();
  if (trayBounds) {
    const x = Math.round(trayBounds.x + trayBounds.width / 2 - windowBounds.width / 2);
    const y = Math.round(trayBounds.y + trayBounds.height);
    mainWindow.setPosition(x, y);
  }
  mainWindow.show();
  mainWindow.focus();
}

async function startProcessing(): Promise<void> {
  try {
    const hasPermission = await checkMicrophonePermission();
    if (!hasPermission) {
      mainWindow?.webContents.send('error', { message: 'Microphone permission denied' });
      return;
    }
    await audioProcessor.start();
    mainWindow?.webContents.send('processing-status', { isRunning: true });
  } catch (error) {
    console.error('Failed to start processing:', error);
    mainWindow?.webContents.send('error', { message: String(error) });
  }
}

async function stopProcessing(): Promise<void> {
  try {
    audioProcessor.stop();
    mainWindow?.webContents.send('processing-status', { isRunning: false });
  } catch (error) {
    console.error('Failed to stop processing:', error);
  }
}

function setupIpcHandlers(): void {
  ipcMain.handle('get-devices', () => {
    deviceManager.refreshDevices();
    return {
      input: deviceManager.getInputDevices(),
      output: deviceManager.getOutputDevices(),
      all: deviceManager.getAllDevices()
    };
  });

  ipcMain.handle('get-config', () => audioProcessor.getConfig());
  ipcMain.handle('get-levels', () => audioProcessor.getLevels());
  ipcMain.handle('get-status', () => ({ isRunning: audioProcessor.getIsRunning() }));
  ipcMain.handle('start-processing', async () => { await startProcessing(); return { success: true }; });
  ipcMain.handle('stop-processing', async () => { await stopProcessing(); return { success: true }; });
  
  ipcMain.handle('set-input-device', (_event, deviceId: number) => {
    audioProcessor.updateConfig({ inputDeviceId: deviceId });
    return { success: true };
  });
  ipcMain.handle('set-reference-device', (_event, deviceId: number | null) => {
    audioProcessor.updateConfig({ referenceDeviceId: deviceId });
    return { success: true };
  });
  ipcMain.handle('set-output-device', (_event, deviceId: number | null) => {
    audioProcessor.updateConfig({ outputDeviceId: deviceId });
    return { success: true };
  });
  ipcMain.handle('set-input-gain', (_event, gain: number) => {
    audioProcessor.updateConfig({ inputGain: gain });
    return { success: true };
  });
  ipcMain.handle('set-reference-gain', (_event, gain: number) => {
    audioProcessor.updateConfig({ referenceGain: gain });
    return { success: true };
  });
  ipcMain.handle('set-step-size', () => ({ success: true }));
  ipcMain.handle('set-reference-delay', () => ({ success: true }));
  ipcMain.handle('reset-filter', () => ({ success: true }));
  
  ipcMain.handle('start-recording', () => {
    audioProcessor.startRecording();
    return { success: true };
  });
  ipcMain.handle('stop-recording', async () => {
    const result = await audioProcessor.stopRecording();
    return { success: true, files: result };
  });
  ipcMain.handle('get-recording-status', () => ({ isRecording: audioProcessor.getIsRecording() }));
  ipcMain.handle('hide-window', () => { mainWindow?.hide(); return { success: true }; });
}

app.whenReady().then(() => {
  deviceManager = getDeviceManager();
  audioProcessor = getSpeexProcessor();

  audioProcessor.on('levels', (levels) => {
    mainWindow?.webContents.send('levels-update', levels);
  });

  audioProcessor.on('error', (error) => {
    mainWindow?.webContents.send('error', error);
  });

  setupIpcHandlers();
  createTray();
  createWindow();
  
  setTimeout(() => showWindow(), 500);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  audioProcessor?.stop();
});

if (process.platform === 'darwin') {
  app.dock?.hide();
}

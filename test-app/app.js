/**
 * Audio Separation Lab - Main Application
 */

// State
let inputAudioBuffer = null;
let referenceAudioBuffer = null;
let audioContext = null;
let selectedAlgorithms = new Set(['nlms']);
let algorithmParams = {};
let results = [];
let currentlyPlaying = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  
  // Merge all algorithm sources
  window.allAlgorithms = {
    ...window.algorithms,
    ...window.advancedAlgorithms,
    ...window.deepLearningAlgorithms
  };
  
  setupDropZones();
  setupAlgorithmCards();
  setupProcessButton();
  initializeParams();
});

// ============================================================
// File Upload Handling
// ============================================================
function setupDropZones() {
  setupDropZone('inputDropZone', 'inputFile', 'inputFilename', 'inputWaveform', (buffer) => {
    inputAudioBuffer = buffer;
    updateProcessButton();
  });
  
  setupDropZone('referenceDropZone', 'referenceFile', 'referenceFilename', 'referenceWaveform', (buffer) => {
    referenceAudioBuffer = buffer;
    updateProcessButton();
  });
}

function setupDropZone(zoneId, inputId, filenameId, waveformId, onLoad) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const filenameEl = document.getElementById(filenameId);
  const waveformContainer = document.getElementById(waveformId);
  
  zone.addEventListener('click', () => input.click());
  
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  
  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });
  
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, zone, filenameEl, waveformContainer, onLoad);
  });
  
  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file, zone, filenameEl, waveformContainer, onLoad);
  });
}

async function handleFile(file, zone, filenameEl, waveformContainer, onLoad) {
  zone.classList.add('has-file');
  filenameEl.textContent = file.name;
  
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    onLoad(audioBuffer);
    
    // Draw waveform
    waveformContainer.classList.add('visible');
    drawWaveform(waveformContainer.querySelector('canvas'), audioBuffer);
    
  } catch (error) {
    console.error('Error loading audio file:', error);
    filenameEl.textContent = 'Error loading file';
    zone.classList.remove('has-file');
  }
}

function drawWaveform(canvas, audioBuffer) {
  const ctx = canvas.getContext('2d');
  const width = canvas.offsetWidth * window.devicePixelRatio;
  const height = canvas.offsetHeight * window.devicePixelRatio;
  
  canvas.width = width;
  canvas.height = height;
  
  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  
  ctx.fillStyle = '#14141c';
  ctx.fillRect(0, 0, width, height);
  
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, '#00e5cc');
  gradient.addColorStop(1, '#e930ff');
  
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1;
  ctx.beginPath();
  
  const centerY = height / 2;
  
  for (let i = 0; i < width; i++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const idx = i * step + j;
      if (idx < data.length) {
        if (data[idx] < min) min = data[idx];
        if (data[idx] > max) max = data[idx];
      }
    }
    
    const y1 = centerY + min * centerY * 0.9;
    const y2 = centerY + max * centerY * 0.9;
    
    ctx.moveTo(i, y1);
    ctx.lineTo(i, y2);
  }
  
  ctx.stroke();
}

// ============================================================
// Algorithm Selection
// ============================================================
function setupAlgorithmCards() {
  const cards = document.querySelectorAll('.algorithm-card');
  
  cards.forEach(card => {
    card.addEventListener('click', () => {
      const algo = card.dataset.algo;
      
      if (selectedAlgorithms.has(algo)) {
        selectedAlgorithms.delete(algo);
        card.classList.remove('selected');
      } else {
        selectedAlgorithms.add(algo);
        card.classList.add('selected');
      }
      
      updateParamsSection();
      updateProcessButton();
    });
  });
}

function initializeParams() {
  for (const [algoId, algo] of Object.entries(window.allAlgorithms)) {
    algorithmParams[algoId] = {};
    for (const [paramId, param] of Object.entries(algo.defaultParams)) {
      algorithmParams[algoId][paramId] = param.value;
    }
  }
  updateParamsSection();
}

function updateParamsSection() {
  const paramsGrid = document.getElementById('paramsGrid');
  paramsGrid.innerHTML = '';
  
  selectedAlgorithms.forEach(algoId => {
    const algo = window.allAlgorithms[algoId];
    if (!algo) return;
    
    for (const [paramId, param] of Object.entries(algo.defaultParams)) {
      const group = document.createElement('div');
      group.className = 'param-group';
      
      const label = document.createElement('label');
      label.textContent = `${algo.name} - ${param.label}`;
      
      const input = document.createElement('input');
      input.type = 'range';
      input.min = param.min;
      input.max = param.max;
      input.step = param.step;
      input.value = algorithmParams[algoId][paramId];
      
      const valueDisplay = document.createElement('span');
      valueDisplay.className = 'param-value';
      valueDisplay.textContent = algorithmParams[algoId][paramId];
      
      input.addEventListener('input', () => {
        algorithmParams[algoId][paramId] = parseFloat(input.value);
        valueDisplay.textContent = parseFloat(input.value).toFixed(
          param.step < 0.01 ? 4 : param.step < 1 ? 2 : 0
        );
      });
      
      group.appendChild(label);
      group.appendChild(input);
      group.appendChild(valueDisplay);
      paramsGrid.appendChild(group);
    }
  });
}

// ============================================================
// Processing
// ============================================================
function setupProcessButton() {
  const btn = document.getElementById('processBtn');
  btn.addEventListener('click', processAudio);
}

function updateProcessButton() {
  const btn = document.getElementById('processBtn');
  btn.disabled = !inputAudioBuffer || !referenceAudioBuffer || selectedAlgorithms.size === 0;
}

async function processAudio() {
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingText = document.getElementById('loadingText');
  
  loadingOverlay.classList.add('visible');
  results = [];
  
  // Get audio data
  const inputData = inputAudioBuffer.getChannelData(0);
  const referenceData = referenceAudioBuffer.getChannelData(0);
  
  // Resample if needed to match lengths
  const minLength = Math.min(inputData.length, referenceData.length);
  const input = new Float32Array(inputData.slice(0, minLength));
  const reference = new Float32Array(referenceData.slice(0, minLength));
  
  const algosArray = Array.from(selectedAlgorithms);
  
  for (let i = 0; i < algosArray.length; i++) {
    const algoId = algosArray[i];
    const algo = window.allAlgorithms[algoId];
    if (!algo) continue;
    
    loadingText.textContent = `Processing ${algo.name} (${i + 1}/${algosArray.length})...`;
    
    // Allow UI to update
    await new Promise(r => setTimeout(r, 50));
    
    try {
      const startTime = performance.now();
      
      // Create filter with current params - handle async algorithms
      let filter;
      if (algo.async) {
        loadingText.textContent = `Loading ${algo.name} model...`;
        await new Promise(r => setTimeout(r, 10));
        filter = await algo.create(algorithmParams[algoId]);
      } else {
        filter = algo.create(algorithmParams[algoId]);
      }
      
      loadingText.textContent = `Processing with ${algo.name}...`;
      await new Promise(r => setTimeout(r, 10));
      
      // Process
      const output = filter.process(input, reference);
      
      const endTime = performance.now();
      const processingTime = endTime - startTime;
      
      // Calculate metrics
      const metrics = calculateMetrics(input, reference, output);
      
      // Create audio buffer for playback
      const outputBuffer = audioContext.createBuffer(1, output.length, audioContext.sampleRate);
      outputBuffer.getChannelData(0).set(output);
      
      results.push({
        algoId,
        name: algo.name,
        output,
        outputBuffer,
        metrics,
        processingTime
      });
      
      // Clean up if the filter has a destroy method
      if (filter.destroy) {
        filter.destroy();
      }
      
    } catch (error) {
      console.error(`Error processing ${algo.name}:`, error);
      results.push({
        algoId,
        name: algo.name + ' (ERROR)',
        output: new Float32Array(input.length),
        outputBuffer: audioContext.createBuffer(1, input.length, audioContext.sampleRate),
        metrics: { suppression: 'N/A', correlation: 'N/A', outputLevel: 'N/A' },
        processingTime: 0,
        error: error.message
      });
    }
  }
  
  loadingOverlay.classList.remove('visible');
  displayResults();
}

function calculateMetrics(input, reference, output) {
  // Calculate RMS values
  const inputRMS = Math.sqrt(input.reduce((s, x) => s + x*x, 0) / input.length);
  const refRMS = Math.sqrt(reference.reduce((s, x) => s + x*x, 0) / reference.length);
  const outputRMS = Math.sqrt(output.reduce((s, x) => s + x*x, 0) / output.length);
  
  // Suppression ratio (how much of reference was removed)
  const suppression = refRMS > 0 ? 20 * Math.log10(outputRMS / refRMS) : 0;
  
  // Signal preservation (correlation with input voice)
  let correlation = 0;
  for (let i = 0; i < output.length; i++) {
    correlation += output[i] * input[i];
  }
  correlation /= (outputRMS * inputRMS * output.length);
  
  // SNR estimate
  const snr = inputRMS > 0 ? 20 * Math.log10(outputRMS / inputRMS) : 0;
  
  return {
    suppression: suppression.toFixed(1),
    correlation: (correlation * 100).toFixed(1),
    snr: snr.toFixed(1),
    outputLevel: (outputRMS * 100).toFixed(2)
  };
}

// ============================================================
// Results Display
// ============================================================
function displayResults() {
  const resultsSection = document.getElementById('resultsSection');
  const resultsGrid = document.getElementById('resultsGrid');
  
  resultsSection.classList.add('visible');
  resultsGrid.innerHTML = '';
  
  results.forEach((result, index) => {
    const card = createResultCard(result, index);
    resultsGrid.appendChild(card);
  });
  
  updateComparisonChart();
}

function createResultCard(result, index) {
  const card = document.createElement('div');
  card.className = 'result-card';
  
  card.innerHTML = `
    <h4>
      ${result.name}
      <span style="font-size: 0.75rem; color: var(--text-muted);">${result.processingTime.toFixed(0)}ms</span>
    </h4>
    <div class="metrics">
      <div class="metric">
        <div class="value">${result.metrics.suppression}dB</div>
        <div class="label">Suppression</div>
      </div>
      <div class="metric">
        <div class="value">${result.metrics.correlation}%</div>
        <div class="label">Correlation</div>
      </div>
      <div class="metric">
        <div class="value">${result.metrics.outputLevel}%</div>
        <div class="label">Output Level</div>
      </div>
    </div>
    <div class="result-waveform">
      <canvas id="waveform-${index}"></canvas>
    </div>
    <div class="audio-controls">
      <button class="play-btn" data-index="${index}">▶ Play Output</button>
      <button class="download-btn" data-index="${index}">💾 Download</button>
    </div>
  `;
  
  // Draw waveform after adding to DOM
  setTimeout(() => {
    const canvas = document.getElementById(`waveform-${index}`);
    if (canvas) {
      drawOutputWaveform(canvas, result.output);
    }
  }, 0);
  
  // Setup play button
  const playBtn = card.querySelector('.play-btn');
  playBtn.addEventListener('click', () => playResult(result, playBtn));
  
  // Setup download button
  const downloadBtn = card.querySelector('.download-btn');
  downloadBtn.addEventListener('click', () => downloadResult(result));
  
  return card;
}

function drawOutputWaveform(canvas, data) {
  const ctx = canvas.getContext('2d');
  const width = canvas.offsetWidth * window.devicePixelRatio;
  const height = canvas.offsetHeight * window.devicePixelRatio;
  
  canvas.width = width;
  canvas.height = height;
  
  const step = Math.ceil(data.length / width);
  
  ctx.fillStyle = '#14141c';
  ctx.fillRect(0, 0, width, height);
  
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, '#ff6b35');
  gradient.addColorStop(1, '#00e5cc');
  
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1;
  ctx.beginPath();
  
  const centerY = height / 2;
  
  for (let i = 0; i < width; i++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const idx = i * step + j;
      if (idx < data.length) {
        if (data[idx] < min) min = data[idx];
        if (data[idx] > max) max = data[idx];
      }
    }
    
    const y1 = centerY + min * centerY * 0.9;
    const y2 = centerY + max * centerY * 0.9;
    
    ctx.moveTo(i, y1);
    ctx.lineTo(i, y2);
  }
  
  ctx.stroke();
}

function playResult(result, button) {
  // Stop any currently playing audio
  if (currentlyPlaying) {
    currentlyPlaying.source.stop();
    currentlyPlaying.button.classList.remove('playing');
    currentlyPlaying.button.textContent = '▶ Play Output';
    
    if (currentlyPlaying.button === button) {
      currentlyPlaying = null;
      return;
    }
  }
  
  const source = audioContext.createBufferSource();
  source.buffer = result.outputBuffer;
  source.connect(audioContext.destination);
  source.start();
  
  button.classList.add('playing');
  button.textContent = '⏹ Stop';
  
  currentlyPlaying = { source, button };
  
  source.onended = () => {
    button.classList.remove('playing');
    button.textContent = '▶ Play Output';
    currentlyPlaying = null;
  };
}

function downloadResult(result) {
  // Create WAV file
  const wavBuffer = createWavFile(result.output, audioContext.sampleRate);
  const blob = new Blob([wavBuffer], { type: 'audio/wav' });
  
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${result.algoId}_output.wav`;
  link.click();
  
  URL.revokeObjectURL(link.href);
}

function createWavFile(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  
  // Convert float to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  
  return buffer;
}

function updateComparisonChart() {
  const chart = document.getElementById('comparisonChart');
  chart.innerHTML = '';
  
  // Find max suppression for scaling
  const suppressions = results.map(r => Math.abs(parseFloat(r.metrics.suppression)));
  const maxSuppression = Math.max(...suppressions, 1);
  
  results.forEach(result => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    
    const suppression = Math.abs(parseFloat(result.metrics.suppression));
    const height = (suppression / maxSuppression) * 140;
    
    bar.innerHTML = `
      <div class="bar" style="height: ${height}px;"></div>
      <div class="name">${result.name}<br><small>${result.metrics.suppression}dB</small></div>
    `;
    
    chart.appendChild(bar);
  });
}


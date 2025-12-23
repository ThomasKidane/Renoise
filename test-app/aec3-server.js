#!/usr/bin/env node
/**
 * AEC3 Processing Server
 * Runs WebRTC AEC3 on uploaded audio files
 * 
 * IMPORTANT: AEC3 demo expects 16-bit PCM WAV files!
 * This server automatically converts input files to the correct format.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const PORT = 8081;
const AEC3_DEMO = path.join(__dirname, '../vendor/aec3/build/aec3_demo');
const TEMP_DIR = path.join(__dirname, 'temp');

// Ensure temp directory exists
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}

/**
 * Convert WAV file to 16-bit PCM format required by AEC3
 * @param {string} inputPath - Input WAV file path
 * @param {string} outputPath - Output 16-bit PCM WAV file path
 * @returns {boolean} - Success status
 */
function convertTo16BitPCM(inputPath, outputPath) {
  try {
    // Use ffmpeg to convert to 16-bit PCM at same sample rate
    execSync(`ffmpeg -y -i "${inputPath}" -acodec pcm_s16le "${outputPath}" 2>/dev/null`, {
      timeout: 30000,
      stdio: 'pipe'
    });
    return true;
  } catch (e) {
    console.error(`FFmpeg conversion failed: ${e.message}`);
    return false;
  }
}

const server = http.createServer(async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  if (req.method === 'POST' && req.url === '/process') {
    let body = [];
    
    req.on('data', chunk => body.push(chunk));
    req.on('end', async () => {
      try {
        const data = JSON.parse(Buffer.concat(body).toString());
        const { reference, input } = data;
        
        // Decode base64 audio data
        const refBuffer = Buffer.from(reference, 'base64');
        const inputBuffer = Buffer.from(input, 'base64');
        
        // Write temp files (original format)
        const timestamp = Date.now();
        const refPathOrig = path.join(TEMP_DIR, `ref_orig_${timestamp}.wav`);
        const inputPathOrig = path.join(TEMP_DIR, `input_orig_${timestamp}.wav`);
        const refPath = path.join(TEMP_DIR, `ref_${timestamp}.wav`);
        const inputPath = path.join(TEMP_DIR, `input_${timestamp}.wav`);
        const outputPath = path.join(TEMP_DIR, `output_${timestamp}.wav`);
        
        fs.writeFileSync(refPathOrig, refBuffer);
        fs.writeFileSync(inputPathOrig, inputBuffer);
        
        console.log(`Processing with AEC3...`);
        console.log(`  Reference (orig): ${refPathOrig} (${refBuffer.length} bytes)`);
        console.log(`  Input (orig): ${inputPathOrig} (${inputBuffer.length} bytes)`);
        
        // Convert to 16-bit PCM (AEC3 requirement!)
        console.log(`  Converting to 16-bit PCM...`);
        if (!convertTo16BitPCM(refPathOrig, refPath)) {
          throw new Error('Failed to convert reference to 16-bit PCM');
        }
        if (!convertTo16BitPCM(inputPathOrig, inputPath)) {
          throw new Error('Failed to convert input to 16-bit PCM');
        }
        
        console.log(`  Reference (16-bit): ${refPath}`);
        console.log(`  Input (16-bit): ${inputPath}`);
        
        // Run AEC3
        try {
          execSync(`"${AEC3_DEMO}" "${refPath}" "${inputPath}" "${outputPath}"`, {
            timeout: 60000,
            stdio: 'pipe'
          });
        } catch (e) {
          console.error('AEC3 error:', e.message);
          throw new Error('AEC3 processing failed');
        }
        
        // Read output
        if (!fs.existsSync(outputPath)) {
          throw new Error('AEC3 did not produce output file');
        }
        
        const outputBuffer = fs.readFileSync(outputPath);
        const outputBase64 = outputBuffer.toString('base64');
        
        // Cleanup all temp files
        try {
          fs.unlinkSync(refPathOrig);
          fs.unlinkSync(inputPathOrig);
          fs.unlinkSync(refPath);
          fs.unlinkSync(inputPath);
          fs.unlinkSync(outputPath);
          // Also cleanup linear.wav that AEC3 creates
          const linearPath = path.join(process.cwd(), 'linear.wav');
          if (fs.existsSync(linearPath)) fs.unlinkSync(linearPath);
        } catch (e) {}
        
        console.log(`  Output: ${outputBuffer.length} bytes`);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
          success: true, 
          output: outputBase64,
          size: outputBuffer.length
        }));
        
      } catch (error) {
        console.error('Processing error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: error.message }));
      }
    });
    
  } else if (req.method === 'GET' && req.url === '/health') {
    // Check if AEC3 demo exists
    const aec3Exists = fs.existsSync(AEC3_DEMO);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'ok', 
      aec3Available: aec3Exists,
      aec3Path: AEC3_DEMO
    }));
    
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`AEC3 Processing Server running on http://localhost:${PORT}`);
  console.log(`AEC3 demo path: ${AEC3_DEMO}`);
  console.log(`AEC3 exists: ${fs.existsSync(AEC3_DEMO)}`);
});


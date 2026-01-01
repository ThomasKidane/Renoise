#!/bin/bash
# Setup script for SepFormer (SpeechBrain)
# This creates a Python virtual environment with the required dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up SepFormer for Renoise ==="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python 3 not found. Please install Python 3.10 or 3.11"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (compatible version for SpeechBrain)
echo ""
echo "Installing PyTorch..."
pip install 'torch>=2.0,<2.5' 'torchaudio>=2.0,<2.5'

# Install SpeechBrain and dependencies
echo ""
echo "Installing SpeechBrain..."
pip install speechbrain requests soundfile 'huggingface_hub<0.25'

# Test the installation
echo ""
echo "Testing installation..."
python -c "
from speechbrain.inference.separation import SepformerSeparation
print('SpeechBrain SepFormer loaded successfully!')
print('The model will be downloaded on first use (~200MB)')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test SepFormer manually:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python sepformer_process.py --input <input.wav> --output <output.wav>"
echo ""
echo "The app will automatically use SepFormer for recording post-processing."


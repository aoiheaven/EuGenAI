#!/bin/bash

# Setup script for Medical Multimodal Chain-of-Thought Project
# This script will set up the environment and create necessary directories

echo "================================================"
echo "Medical Multimodal CoT - Setup Script"
echo "================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your terminal and run this script again."
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate    (Linux/Mac)"
echo "  .venv\\Scripts\\activate      (Windows)"

# Create necessary directories
echo ""
echo "Creating project directories..."

mkdir -p data/images
mkdir -p data/train
mkdir -p data/val
mkdir -p data/test
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs

echo "✓ Directories created:"
echo "  - data/images    (for medical images)"
echo "  - data/train     (for training data)"
echo "  - data/val       (for validation data)"
echo "  - data/test      (for test data)"
echo "  - checkpoints    (for model checkpoints)"
echo "  - logs           (for training logs)"
echo "  - outputs        (for inference outputs)"

# Instructions
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Install dependencies:"
echo "   uv pip install -e ."
echo ""
echo "3. Prepare your data in JSON format (see data_format_example.json)"
echo ""
echo "4. Start training:"
echo "   python src/train.py --config configs/default_config.yaml"
echo ""
echo "For more information, see README.md"
echo "================================================"


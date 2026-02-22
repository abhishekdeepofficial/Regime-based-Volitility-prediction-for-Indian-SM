#!/bin/bash
# ============================================
# Lightning.ai Studio Setup Script
# Run this ONCE after uploading the project
# ============================================

echo "=== Setting up RResearch Project on Lightning.ai ==="

# 1. Install dependencies
echo ">>> Installing Python dependencies..."
pip install pandas numpy matplotlib seaborn scipy arch statsmodels scikit-learn hmmlearn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-forecasting lightning tensorboard optuna

# 2. Create necessary directories
echo ">>> Creating directories..."
mkdir -p data/processed
mkdir -p checkpoints/tft
mkdir -p reports/figures
mkdir -p lightning_logs

# 3. Verify GPU
echo ">>> Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

echo ""
echo "=== Setup complete! ==="
echo "Now run: python run_pipeline.py"

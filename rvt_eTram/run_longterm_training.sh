#!/bin/bash
# Long-term Training Script for Screen Session

echo "🚀 Starting 4-scale FPN Long-term Training in Screen Session"
echo "📅 Started at: $(date)"

# Change to working directory
cd /home/oeoiewt/eTraM/rvt_eTram

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rvt

# Verify environment
echo "🐍 Python: $(which python)"
echo "📦 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the long-term training
python longterm_training.py

echo "✅ Training session completed at: $(date)"

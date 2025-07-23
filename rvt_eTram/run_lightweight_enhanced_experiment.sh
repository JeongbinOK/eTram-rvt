#!/bin/bash
# Lightweight Enhanced ConvLSTM Training Script
# Phase 1 Experiment: Small Object Detection Enhancement
# Expected: 14.7% parameter overhead, 19-20% small objects mAP

set -e  # Exit on any error

# Configuration
EXPERIMENT_ID="lightweight_enhanced_100k"
DATA_DIR="/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample"
GPU_ID=0

echo "🚀 Starting Lightweight Enhanced ConvLSTM Experiment"
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Data Directory: ${DATA_DIR}"
echo "Expected Performance: 35-36% overall mAP, 19-20% small objects mAP"

# Create experiment directory structure
echo "📁 Creating experiment directory structure..."
mkdir -p experiments/${EXPERIMENT_ID}/{checkpoints,confusion_matrices,training_logs,validation_results}

# Backup configuration
echo "💾 Backing up configuration..."
cp config/model/maxvit_yolox/lightweight_enhanced.yaml experiments/${EXPERIMENT_ID}/model_config.yaml
cp config/experiment/gen4/lightweight_enhanced_100k.yaml experiments/${EXPERIMENT_ID}/experiment_config.yaml

# Save training command for reproducibility
cat > experiments/${EXPERIMENT_ID}/training_command.txt << EOF
# Lightweight Enhanced ConvLSTM Training Command
# Date: $(date)
# Phase 1 Implementation with 14.7% parameter overhead

python train.py \\
  model=rnndet \\
  dataset=gen4 \\
  dataset.path=${DATA_DIR} \\
  +experiment/gen4="lightweight_enhanced_100k.yaml" \\
  hardware.gpus=${GPU_ID} \\
  batch_size.train=6 \\
  batch_size.eval=2 \\
  hardware.num_workers.train=4 \\
  hardware.num_workers.eval=3 \\
  training.max_steps=100000 \\
  dataset.train.sampling=stream \\
  +model.head.num_classes=8 \\
  wandb.project_name=etram_enhanced_convlstm \\
  wandb.group_name=${EXPERIMENT_ID}
EOF

echo "📋 Training command saved to experiments/${EXPERIMENT_ID}/training_command.txt"

# Display training instructions
echo ""
echo "🎯 TRAINING INSTRUCTIONS (Following CLAUDE.md Standard Process):"
echo ""
echo "1. Manual Training Execution (Recommended):"
echo "   screen -dmS ${EXPERIMENT_ID}"
echo "   screen -S ${EXPERIMENT_ID} -p 0 -X stuff \"cd /home/oeoiewt/eTraM/rvt_eTram\n\""
echo "   screen -S ${EXPERIMENT_ID} -p 0 -X stuff \"$(cat experiments/${EXPERIMENT_ID}/training_command.txt | grep -v '^#' | tr '\n' ' ' | tr '\\' ' '); echo 'Training completed! Press Enter to continue...'; read\n\""
echo ""
echo "2. Check Progress:"
echo "   screen -r ${EXPERIMENT_ID}"
echo ""
echo "3. After Training - Validation:"
echo "   python validation.py dataset=gen4 dataset.path=${DATA_DIR} \\"
echo "     checkpoint=experiments/${EXPERIMENT_ID}/checkpoints/final_model.ckpt \\"
echo "     +experiment/gen4=\"lightweight_enhanced_100k.yaml\" \\"
echo "     hardware.gpus=${GPU_ID} batch_size.eval=8 +model.head.num_classes=8"
echo ""
echo "🎯 SUCCESS CRITERIA:"
echo "   - Overall mAP: 35-36% (vs 34.02% baseline)"
echo "   - Small objects mAP: 19-20% (vs 17.28% baseline)"
echo "   - Parameter overhead: ~15% (actual: 14.7%)"
echo "   - Memory overhead: <5%"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Use screen for long training sessions"
echo "   - Monitor GPU memory usage"
echo "   - Check training logs regularly"
echo "   - Backup results after completion"

# Ask user if they want to start training
echo ""
read -p "🚀 Start training now in screen session? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting training in screen session: ${EXPERIMENT_ID}"
    screen -dmS ${EXPERIMENT_ID}
    screen -S ${EXPERIMENT_ID} -p 0 -X stuff "cd /home/oeoiewt/eTraM/rvt_eTram\n"
    
    # Construct and send training command
    TRAINING_CMD=$(cat experiments/${EXPERIMENT_ID}/training_command.txt | grep -v '^#' | tr '\n' ' ' | tr '\\' ' ')
    screen -S ${EXPERIMENT_ID} -p 0 -X stuff "${TRAINING_CMD}; echo 'Training completed! Press Enter to continue...'; read\n"
    
    echo "✅ Training started in screen session: ${EXPERIMENT_ID}"
    echo "📺 To monitor progress: screen -r ${EXPERIMENT_ID}"
    echo "🔄 To detach from screen: Ctrl+A, then D"
else
    echo "👍 Training not started. Use the commands above to start manually."
fi

echo ""
echo "📝 This experiment implements Phase 1 of the ConvLSTM enhancement plan:"
echo "   - Lightweight Enhanced ConvLSTM with temporal attention"
echo "   - Event-density adaptive processing"
echo "   - Small object detection optimization"
echo "   - 14.7% parameter overhead (target achieved)"
echo ""
echo "📊 Expected Results Timeline:"
echo "   - Training: ~5-6 hours"
echo "   - Validation: ~10 minutes"
echo "   - Results analysis: ~20 minutes"
echo ""
echo "🎉 Phase 1 implementation ready for training!"
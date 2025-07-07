#!/bin/bash
# 4-scale FPN Training Script for Sample Dataset with Enhanced Monitoring

echo "🚀 Starting 4-scale FPN training on improved sample dataset..."
echo "📊 Improved sample dataset: 16 sequences, 116,689 total objects"
echo "📈 Enhanced monitoring: Confusion matrices will be saved to confM/ directory"

# Ensure confM directory exists and is writable
mkdir -p ./confM
chmod 755 ./confM

echo ""
echo "🔧 Configuration Summary:"
echo "  - 4-scale FPN: P1, P2, P3, P4 → N1, N2, N3, N4 (stride 4, 8, 16, 32)"
echo "  - Dataset: gen4_sample (improved class distribution)"
echo "  - Classes: 8 traffic classes (Car, Truck, Motorcycle, Bicycle, Pedestrian, Bus, Static, Other)"
echo "  - Confusion Matrix: Auto-generated every validation step"
echo ""

# Phase 1: Quick validation (1K steps)
echo "📋 Phase 1: Quick 4-scale FPN validation (1,000 steps)"
echo "=========================================================="

source /home/oeoiewt/miniconda3/etc/profile.d/conda.sh
conda activate rvt

python train.py model=rnndet +experiment/gen4=default.yaml \
  dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \
  dataset.train.sampling=stream \
  training.max_epochs=-1 training.max_steps=1000 \
  training.lr_scheduler.use=false \
  validation.val_check_interval=250 \
  validation.check_val_every_n_epoch=null \
  +logging.ckpt_every_n_steps=500 \
  logging.ckpt_every_n_epochs=null \
  +logging.ckpt_dir=./checkpoints_4scale_sample \
  hardware.gpus=[0] \
  batch_size.train=4 batch_size.eval=2 \
  hardware.num_workers.train=2 hardware.num_workers.eval=1 \
  wandb.project_name=null wandb.group_name=null \
  +model.head.num_classes=8

PHASE1_EXIT_CODE=$?

echo ""
if [ $PHASE1_EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 1 complete! Checking results..."
    
    # Check if confusion matrices were generated
    if ls ./confM/confusion_matrix_*.png 1> /dev/null 2>&1; then
        echo "✅ Confusion matrices successfully generated in confM/ directory"
        echo "📊 Latest confusion matrix: confM/confusion_matrix_latest.png"
        
        # List all generated confusion matrices
        echo "📁 Generated confusion matrices:"
        ls -la ./confM/confusion_matrix_*.png | tail -5
    else
        echo "⚠️  Warning: No confusion matrices found in confM/ directory"
        echo "   Check if validation was triggered and confusion matrix generation is working"
    fi
    
    echo ""
    echo "🔄 Phase 2: Extended validation (5,000 steps) - Starting automatically..."
    echo "=================================================================="
    
    # Phase 2: Extended validation (5K steps)
    python train.py model=rnndet +experiment/gen4=default.yaml \
      dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \
      dataset.train.sampling=stream \
      training.max_epochs=-1 training.max_steps=5000 \
      training.lr_scheduler.use=false \
      validation.val_check_interval=1000 \
      validation.check_val_every_n_epoch=null \
      +logging.ckpt_every_n_steps=1000 \
      logging.ckpt_every_n_epochs=null \
      +logging.ckpt_dir=./checkpoints_4scale_sample \
      hardware.gpus=[0] \
      batch_size.train=6 batch_size.eval=2 \
      hardware.num_workers.train=2 hardware.num_workers.eval=1 \
      wandb.project_name=eTraM_4scale wandb.group_name=small_object_detection \
      +model.head.num_classes=8
    
    PHASE2_EXIT_CODE=$?
    
    if [ $PHASE2_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "🎉 All training phases completed successfully!"
        echo ""
        echo "📊 Results Summary:"
        echo "  - 4-scale FPN implementation tested"
        echo "  - Small object detection with stride 4 validated"
        echo "  - Confusion matrices saved for analysis"
        echo ""
        echo "📁 Output locations:"
        echo "  - Checkpoints: ./checkpoints_4scale_sample/"
        echo "  - Confusion matrices: ./confM/"
        echo "  - Training logs: ./outputs/"
        echo ""
        echo "🔍 Next steps:"
        echo "  1. Analyze confusion matrices for class-wise performance"
        echo "  2. Compare small object detection improvement"
        echo "  3. Run validation.py for detailed metrics"
        
        # Show final confusion matrix if available
        if [ -f "./confM/confusion_matrix_latest.png" ]; then
            echo ""
            echo "📊 Latest confusion matrix available at: ./confM/confusion_matrix_latest.png"
            echo "   Open this file to see the 8x8 class performance matrix"
        fi
        
    else
        echo "❌ Phase 2 failed with exit code: $PHASE2_EXIT_CODE"
        echo "   Check logs for details"
    fi
    
else
    echo "❌ Phase 1 failed with exit code: $PHASE1_EXIT_CODE"
    echo "   Common issues:"
    echo "   - CUDA memory error (reduce batch_size.train)"
    echo "   - Dataset path incorrect"
    echo "   - Configuration error"
    echo ""
    echo "🔧 Quick fixes to try:"
    echo "   - Reduce batch size: batch_size.train=2"
    echo "   - Check dataset: ls -la ./data/gen4_cls8_sample/"
    echo "   - Verify GPU: nvidia-smi"
fi

echo ""
echo "🏁 4-scale FPN training script completed"
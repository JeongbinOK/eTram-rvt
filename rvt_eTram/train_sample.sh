#!/bin/bash
# Training script for 4-scale FPN validation on sample dataset

echo "🚀 Starting 4-scale FPN training on sample dataset..."

# Quick validation (1K steps)
echo "📋 Phase 1: Quick validation (1,000 steps)"
python train.py model=rnndet +experiment/gen4=default.yaml \
  dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \
  dataset.train.sampling=stream \
  training.max_epochs=-1 training.max_steps=1000 \
  training.lr_scheduler.use=false \
  validation.val_check_interval=500 \
  validation.check_val_every_n_epoch=null \
  +logging.ckpt_every_n_steps=500 \
  logging.ckpt_every_n_epochs=null \
  +logging.ckpt_dir=./checkpoints_sample \
  hardware.gpus=[0] \
  batch_size.train=4 batch_size.eval=2 \
  hardware.num_workers.train=2 hardware.num_workers.eval=1 \
  wandb.project_name=null wandb.group_name=null \
  +model.head.num_classes=8

echo "✅ Quick validation complete!"

# Extended validation (10K steps) - uncomment if initial test passes
# echo "📋 Phase 2: Extended validation (10,000 steps)"
# python train.py model=rnndet +experiment/gen4=default.yaml \
#   dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \
#   dataset.train.sampling=stream \
#   training.max_epochs=-1 training.max_steps=10000 \
#   training.lr_scheduler.use=false \
#   validation.val_check_interval=2000 \
#   validation.check_val_every_n_epoch=null \
#   +logging.ckpt_every_n_steps=2000 \
#   logging.ckpt_every_n_epochs=null \
#   +logging.ckpt_dir=./checkpoints_sample \
#   hardware.gpus=[0] \
#   batch_size.train=6 batch_size.eval=2 \
#   hardware.num_workers.train=2 hardware.num_workers.eval=1 \
#   wandb.project_name=eTraM_4scale wandb.group_name=small_object_detection \
#   +model.head.num_classes=8

echo "🎉 All training phases complete!"

#!/usr/bin/env python3
"""
Create representative sample dataset using selected sequences
"""

import json
import shutil
from pathlib import Path
import os

def create_sample_dataset():
    """Create sample dataset with selected representative sequences."""
    
    # Load selected samples
    with open('/home/oeoiewt/eTraM/rvt_eTram/selected_samples.json', 'r') as f:
        selected_samples = json.load(f)
    
    # Define paths
    source_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8')
    target_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_sample')
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset at: {target_path}")
    print(f"Source dataset: {source_path}")
    
    total_copied = 0
    total_size_mb = 0
    
    for split in ['train', 'val', 'test']:
        if not selected_samples[split]:
            continue
            
        print(f"\n📁 Processing {split} split...")
        
        for i, sample in enumerate(selected_samples[split]):
            seq_name = sample['sequence_name']
            source_seq_path = source_path / split / seq_name
            target_seq_path = target_path / split / seq_name
            
            if not source_seq_path.exists():
                print(f"  ⚠️  Warning: {source_seq_path} does not exist")
                continue
            
            # Copy the entire sequence directory
            print(f"  {i+1:2d}. Copying {seq_name}...")
            
            try:
                # Use symbolic links to save space (comment out if you want full copies)
                # shutil.copytree(source_seq_path, target_seq_path, symlinks=True)
                
                # Alternative: Create symbolic links manually for better control
                target_seq_path.mkdir(exist_ok=True)
                
                for subdir in ['event_representations_v2', 'labels_v2']:
                    source_subdir = source_seq_path / subdir
                    target_subdir = target_seq_path / subdir
                    
                    if source_subdir.exists():
                        # Create symbolic link to save disk space
                        if target_subdir.exists():
                            target_subdir.unlink() if target_subdir.is_symlink() else shutil.rmtree(target_subdir)
                        target_subdir.symlink_to(source_subdir.absolute())
                        
                        # Calculate size
                        size_mb = sum(f.stat().st_size for f in source_subdir.rglob('*') if f.is_file()) / (1024*1024)
                        total_size_mb += size_mb
                        print(f"      -> {subdir}: {size_mb:.1f} MB")
                
                total_copied += 1
                
            except Exception as e:
                print(f"  ❌ Error copying {seq_name}: {e}")
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"📊 Summary:")
    print(f"  - Total sequences copied: {total_copied}")
    print(f"  - Total data size: {total_size_mb:.1f} MB")
    print(f"  - Location: {target_path}")
    
    return target_path

def create_sample_config():
    """Create configuration file for sample dataset."""
    
    config_content = '''defaults:
  - base

name: gen4_sample
ev_repr_name: 'stacked_histogram_dt=50_nbins=10'
sequence_length: 5
resolution_hw: [720, 1280]
downsample_by_factor_2: True
only_load_end_labels: False
'''
    
    config_path = Path('/home/oeoiewt/eTraM/rvt_eTram/config/dataset/gen4_sample.yaml')
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n📝 Sample dataset config created: {config_path}")
    return config_path

def print_sample_info():
    """Print information about the created sample dataset."""
    
    with open('/home/oeoiewt/eTraM/rvt_eTram/selected_samples.json', 'r') as f:
        selected_samples = json.load(f)
    
    print(f"\n📋 Sample Dataset Information:")
    print(f"=" * 50)
    
    for split in ['train', 'val', 'test']:
        if not selected_samples[split]:
            continue
            
        print(f"\n{split.upper()} ({len(selected_samples[split])} sequences):")
        total_objects = sum(sample['stats']['total_objects'] for sample in selected_samples[split])
        small_objects = sum(sample['stats']['size_distribution']['small'] for sample in selected_samples[split])
        
        print(f"  Total objects: {total_objects:,}")
        print(f"  Small objects: {small_objects:,} ({small_objects/total_objects*100:.1f}%)")
        
        for sample in selected_samples[split]:
            seq_name = sample['sequence_name']
            stats = sample['stats']
            print(f"    • {seq_name}: {stats['total_objects']:,} objects, {stats['small_ratio']:.3f} small ratio")

def create_training_script():
    """Create optimized training script for sample dataset."""
    
    script_content = '''#!/bin/bash
# Training script for 4-scale FPN validation on sample dataset

echo "🚀 Starting 4-scale FPN training on sample dataset..."

# Quick validation (1K steps)
echo "📋 Phase 1: Quick validation (1,000 steps)"
python train.py model=rnndet +experiment/gen4=default.yaml \\
  dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \\
  dataset.train.sampling=stream \\
  training.max_epochs=-1 training.max_steps=1000 \\
  training.lr_scheduler.use=false \\
  validation.val_check_interval=500 \\
  validation.check_val_every_n_epoch=null \\
  +logging.ckpt_every_n_steps=500 \\
  logging.ckpt_every_n_epochs=null \\
  +logging.ckpt_dir=./checkpoints_sample \\
  hardware.gpus=[0] \\
  batch_size.train=4 batch_size.eval=2 \\
  hardware.num_workers.train=2 hardware.num_workers.eval=1 \\
  wandb.project_name=null wandb.group_name=null \\
  +model.head.num_classes=8

echo "✅ Quick validation complete!"

# Extended validation (10K steps) - uncomment if initial test passes
# echo "📋 Phase 2: Extended validation (10,000 steps)"
# python train.py model=rnndet +experiment/gen4=default.yaml \\
#   dataset=gen4_sample dataset.path=./data/gen4_cls8_sample \\
#   dataset.train.sampling=stream \\
#   training.max_epochs=-1 training.max_steps=10000 \\
#   training.lr_scheduler.use=false \\
#   validation.val_check_interval=2000 \\
#   validation.check_val_every_n_epoch=null \\
#   +logging.ckpt_every_n_steps=2000 \\
#   logging.ckpt_every_n_epochs=null \\
#   +logging.ckpt_dir=./checkpoints_sample \\
#   hardware.gpus=[0] \\
#   batch_size.train=6 batch_size.eval=2 \\
#   hardware.num_workers.train=2 hardware.num_workers.eval=1 \\
#   wandb.project_name=eTraM_4scale wandb.group_name=small_object_detection \\
#   +model.head.num_classes=8

echo "🎉 All training phases complete!"
'''
    
    script_path = Path('/home/oeoiewt/eTraM/rvt_eTram/train_sample.sh')
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"\n🎯 Training script created: {script_path}")
    print(f"   Usage: ./train_sample.sh")

if __name__ == "__main__":
    print("Creating representative sample dataset...")
    
    # Create sample dataset
    sample_path = create_sample_dataset()
    
    # Create configuration
    config_path = create_sample_config()
    
    # Print sample information
    print_sample_info()
    
    # Create training script
    create_training_script()
    
    print(f"\n🎉 Sample dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Review the sample dataset at: {sample_path}")
    print(f"2. Run quick validation: ./train_sample.sh")
    print(f"3. Monitor training for 4-scale FPN performance")
#!/usr/bin/env python3
"""
Comprehensive statistical analysis of gen4_cls8 dataset for representative sampling
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sequence(labels_file_path):
    """Analyze a single sequence and return statistics."""
    try:
        data = np.load(str(labels_file_path))
        labels = data['labels']
        
        if len(labels) == 0:
            return None
            
        # Extract relevant fields
        class_ids = labels['class_id']
        x, y, w, h = labels['x'], labels['y'], labels['w'], labels['h']
        confidences = labels['class_confidence']
        
        # Calculate areas (normalized by image size, assuming 640x480 downsampled)
        areas = (w * h) / (640 * 480)  # Normalize to get relative area
        
        # Small object analysis (area < 0.01 is considered small)
        small_mask = areas < 0.01
        medium_mask = (areas >= 0.01) & (areas < 0.05)
        large_mask = areas >= 0.05
        
        stats = {
            'total_objects': len(labels),
            'class_distribution': np.bincount(class_ids, minlength=8).tolist(),
            'area_stats': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas)),
                'median': float(np.median(areas))
            },
            'size_distribution': {
                'small': int(np.sum(small_mask)),
                'medium': int(np.sum(medium_mask)), 
                'large': int(np.sum(large_mask))
            },
            'small_ratio': float(np.sum(small_mask) / len(labels)),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences))
            }
        }
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {labels_file_path}: {e}")
        return None

def comprehensive_dataset_analysis():
    """Perform comprehensive analysis of the entire dataset."""
    base_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8')
    
    results = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        if not split_path.exists():
            continue
            
        print(f"Analyzing {split} split...")
        
        for seq_dir in sorted(split_path.iterdir()):
            if not seq_dir.is_dir():
                continue
                
            labels_file = seq_dir / 'labels_v2' / 'labels.npz'
            if not labels_file.exists():
                continue
                
            stats = analyze_sequence(labels_file)
            if stats is not None:
                # Add metadata
                stats['sequence_name'] = seq_dir.name
                stats['is_night'] = 'night' in seq_dir.name
                stats['day_type'] = 'night' if 'night' in seq_dir.name else 'day'
                
                results[split][seq_dir.name] = stats
                
        print(f"  Found {len(results[split])} valid sequences in {split}")
    
    return results

def print_dataset_summary(results):
    """Print comprehensive dataset summary."""
    class_names = ['Car', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Bus', 'Static', 'Other']
    
    for split in ['train', 'val', 'test']:
        if not results[split]:
            continue
            
        print(f"\n{'='*50}")
        print(f"{split.upper()} SPLIT ANALYSIS")
        print(f"{'='*50}")
        
        sequences = list(results[split].values())
        
        # Basic statistics
        total_sequences = len(sequences)
        day_sequences = sum(1 for s in sequences if not s['is_night'])
        night_sequences = sum(1 for s in sequences if s['is_night'])
        total_objects = sum(s['total_objects'] for s in sequences)
        
        print(f"Sequences: {total_sequences} (Day: {day_sequences}, Night: {night_sequences})")
        print(f"Total objects: {total_objects:,}")
        
        # Class distribution
        total_class_dist = np.sum([s['class_distribution'] for s in sequences], axis=0)
        print(f"\nClass Distribution:")
        for i, (name, count) in enumerate(zip(class_names, total_class_dist)):
            percentage = count / total_objects * 100 if total_objects > 0 else 0
            print(f"  {i}: {name:<12} {count:>6,} ({percentage:>5.1f}%)")
        
        # Size distribution
        total_small = sum(s['size_distribution']['small'] for s in sequences)
        total_medium = sum(s['size_distribution']['medium'] for s in sequences)
        total_large = sum(s['size_distribution']['large'] for s in sequences)
        
        print(f"\nObject Size Distribution:")
        print(f"  Small objects:  {total_small:>6,} ({total_small/total_objects*100:>5.1f}%)")
        print(f"  Medium objects: {total_medium:>6,} ({total_medium/total_objects*100:>5.1f}%)")
        print(f"  Large objects:  {total_large:>6,} ({total_large/total_objects*100:>5.1f}%)")
        
        # Small object ratios by sequence
        small_ratios = [s['small_ratio'] for s in sequences]
        print(f"\nSmall Object Ratio Statistics:")
        print(f"  Mean: {np.mean(small_ratios):.3f}")
        print(f"  Std:  {np.std(small_ratios):.3f}")
        print(f"  Min:  {np.min(small_ratios):.3f}")
        print(f"  Max:  {np.max(small_ratios):.3f}")

def select_representative_samples(results):
    """Select representative samples using stratified sampling."""
    
    samples = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split in ['train', 'val', 'test']:
        if not results[split]:
            continue
            
        sequences = list(results[split].values())
        
        # Target sample sizes
        if split == 'train':
            target_size = 8
        else:
            target_size = 2
            
        # Create feature matrix for clustering
        features = []
        seq_names = []
        
        for seq in sequences:
            # Feature vector: [small_ratio, total_objects, night_flag, class_diversity]
            class_dist = np.array(seq['class_distribution'])
            class_diversity = np.sum(class_dist > 0)  # Number of classes present
            
            feature_vec = [
                seq['small_ratio'],
                seq['total_objects'] / 1000,  # Normalize object count
                1.0 if seq['is_night'] else 0.0,
                class_diversity / 8.0  # Normalize class diversity
            ]
            
            features.append(feature_vec)
            seq_names.append(seq['sequence_name'])
        
        features = np.array(features)
        
        # Stratified sampling based on small object ratio
        small_ratios = features[:, 0]
        sorted_indices = np.argsort(small_ratios)
        
        # Ensure day/night balance for train split
        if split == 'train':
            day_indices = [i for i in sorted_indices if not sequences[i]['is_night']]
            night_indices = [i for i in sorted_indices if sequences[i]['is_night']]
            
            # Select 4 day, 4 night sequences with diverse small object ratios
            selected_indices = []
            
            # Day sequences
            day_step = len(day_indices) // 4
            for i in range(4):
                idx = day_indices[min(i * day_step, len(day_indices) - 1)]
                selected_indices.append(idx)
            
            # Night sequences  
            night_step = len(night_indices) // 4
            for i in range(4):
                idx = night_indices[min(i * night_step, len(night_indices) - 1)]
                selected_indices.append(idx)
                
        else:
            # For val/test, select diverse samples
            step = len(sorted_indices) // target_size
            selected_indices = [sorted_indices[i * step] for i in range(target_size)]
        
        # Store selected samples
        for idx in selected_indices:
            samples[split].append({
                'sequence_name': seq_names[idx],
                'stats': sequences[idx]
            })
    
    return samples

def print_selected_samples(selected_samples):
    """Print the selected representative samples."""
    
    for split in ['train', 'val', 'test']:
        if not selected_samples[split]:
            continue
            
        print(f"\n{'='*60}")
        print(f"SELECTED {split.upper()} SAMPLES")
        print(f"{'='*60}")
        
        for i, sample in enumerate(selected_samples[split]):
            seq_name = sample['sequence_name']
            stats = sample['stats']
            
            print(f"\n{i+1}. {seq_name}")
            print(f"   Type: {'Night' if stats['is_night'] else 'Day'}")
            print(f"   Objects: {stats['total_objects']:,}")
            print(f"   Small ratio: {stats['small_ratio']:.3f}")
            print(f"   Classes: {np.sum(np.array(stats['class_distribution']) > 0)}/8")
            
            # Top classes
            class_dist = np.array(stats['class_distribution'])
            top_classes = np.argsort(class_dist)[::-1][:3]
            class_names = ['Car', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Bus', 'Static', 'Other']
            print(f"   Top classes: {', '.join([f'{class_names[c]}({class_dist[c]})' for c in top_classes if class_dist[c] > 0])}")

def save_results(results, selected_samples):
    """Save analysis results and selected samples."""
    
    # Save full analysis
    with open('/home/oeoiewt/eTraM/rvt_eTram/dataset_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save selected samples list
    with open('/home/oeoiewt/eTraM/rvt_eTram/selected_samples.json', 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - dataset_analysis.json (full analysis)")
    print(f"  - selected_samples.json (selected samples)")

if __name__ == "__main__":
    print("Starting comprehensive dataset analysis...")
    
    # Perform analysis
    results = comprehensive_dataset_analysis()
    
    # Print summary
    print_dataset_summary(results)
    
    # Select representative samples
    selected_samples = select_representative_samples(results)
    
    # Print selected samples
    print_selected_samples(selected_samples)
    
    # Save results
    save_results(results, selected_samples)
    
    print(f"\n✅ Analysis complete!")
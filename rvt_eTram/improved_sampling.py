#!/usr/bin/env python3
"""
Improved sampling with better class distribution balance
"""

import json
import numpy as np
from pathlib import Path
import shutil

def improved_stratified_sampling():
    """Perform improved sampling with better class distribution balance."""
    
    # Load full analysis
    with open('/home/oeoiewt/eTraM/rvt_eTram/dataset_analysis.json', 'r') as f:
        results = json.load(f)
    
    improved_samples = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split in ['train', 'val', 'test']:
        if not results[split]:
            continue
            
        sequences = list(results[split].values())
        
        # Calculate target class distribution from original dataset
        target_class_dist = np.sum([s['class_distribution'] for s in sequences], axis=0)
        target_class_ratios = target_class_dist / np.sum(target_class_dist)
        
        print(f"\n=== {split.upper()} SPLIT IMPROVED SAMPLING ===")
        print(f"Target class ratios: {[f'{r:.3f}' for r in target_class_ratios]}")
        
        # Target sample sizes
        if split == 'train':
            target_size = 10  # Increase for better representation
        else:
            target_size = 3   # Increase for better balance
            
        # Calculate weighted scores for each sequence
        sequence_scores = []
        for seq in sequences:
            seq_class_dist = np.array(seq['class_distribution'])
            seq_total = np.sum(seq_class_dist)
            
            if seq_total == 0:
                continue
                
            seq_class_ratios = seq_class_dist / seq_total
            
            # Score based on how well this sequence represents the target distribution
            # Use KL divergence (lower is better)
            kl_div = np.sum(target_class_ratios * np.log((target_class_ratios + 1e-10) / (seq_class_ratios + 1e-10)))
            
            # Additional scoring factors
            small_ratio_score = seq['small_ratio']  # Diversity in small objects
            object_count_score = min(seq['total_objects'] / 5000, 1.0)  # Normalized object count
            night_bonus = 0.1 if seq['is_night'] else 0.0  # Slight preference for night data
            
            # Combined score (lower KL divergence is better, so invert it)
            combined_score = 1.0 / (1.0 + kl_div) + 0.3 * small_ratio_score + 0.2 * object_count_score + night_bonus
            
            sequence_scores.append({
                'sequence': seq,
                'score': combined_score,
                'kl_div': kl_div,
                'small_ratio': small_ratio_score
            })
        
        # Sort by score (descending)
        sequence_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top sequences while maintaining day/night balance
        selected = []
        day_count = 0
        night_count = 0
        max_day = target_size // 2 + 1
        max_night = target_size // 2 + 1
        
        for score_info in sequence_scores:
            seq = score_info['sequence']
            if len(selected) >= target_size:
                break
                
            # Check day/night balance
            if seq['is_night'] and night_count < max_night:
                selected.append(score_info)
                night_count += 1
            elif not seq['is_night'] and day_count < max_day:
                selected.append(score_info)
                day_count += 1
            elif len(selected) < target_size:  # Fill remaining spots
                selected.append(score_info)
        
        # Store selected samples
        for score_info in selected:
            improved_samples[split].append({
                'sequence_name': score_info['sequence']['sequence_name'],
                'stats': score_info['sequence'],
                'selection_score': score_info['score'],
                'kl_divergence': score_info['kl_div']
            })
        
        # Print selection summary
        selected_class_dist = np.sum([s['stats']['class_distribution'] for s in improved_samples[split]], axis=0)
        selected_class_ratios = selected_class_dist / np.sum(selected_class_dist)
        
        print(f"Selected {len(improved_samples[split])} sequences:")
        print(f"Selected class ratios: {[f'{r:.3f}' for r in selected_class_ratios]}")
        print(f"Class ratio differences: {[f'{s-t:+.3f}' for s, t in zip(selected_class_ratios, target_class_ratios)]}")
        
        for i, sample in enumerate(improved_samples[split]):
            seq_name = sample['sequence_name']
            stats = sample['stats']
            print(f"  {i+1}. {seq_name}: score={sample['selection_score']:.3f}, "
                  f"objects={stats['total_objects']:,}, small_ratio={stats['small_ratio']:.3f}")
    
    return improved_samples

def create_improved_sample_dataset(improved_samples):
    """Create improved sample dataset."""
    
    # Remove old sample dataset
    old_target_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_sample')
    if old_target_path.exists():
        shutil.rmtree(old_target_path)
    
    # Create new sample dataset
    source_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8')
    target_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_sample')
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Creating improved sample dataset at: {target_path}")
    
    total_copied = 0
    total_size_mb = 0
    
    for split in ['train', 'val', 'test']:
        if not improved_samples[split]:
            continue
            
        print(f"\n📁 Processing {split} split...")
        
        for i, sample in enumerate(improved_samples[split]):
            seq_name = sample['sequence_name']
            source_seq_path = source_path / split / seq_name
            target_seq_path = target_path / split / seq_name
            
            if not source_seq_path.exists():
                print(f"  ⚠️  Warning: {source_seq_path} does not exist")
                continue
            
            print(f"  {i+1:2d}. Copying {seq_name}...")
            
            try:
                target_seq_path.mkdir(exist_ok=True)
                
                for subdir in ['event_representations_v2', 'labels_v2']:
                    source_subdir = source_seq_path / subdir
                    target_subdir = target_seq_path / subdir
                    
                    if source_subdir.exists():
                        if target_subdir.exists():
                            target_subdir.unlink() if target_subdir.is_symlink() else shutil.rmtree(target_subdir)
                        target_subdir.symlink_to(source_subdir.absolute())
                        
                        size_mb = sum(f.stat().st_size for f in source_subdir.rglob('*') if f.is_file()) / (1024*1024)
                        total_size_mb += size_mb
                        print(f"      -> {subdir}: {size_mb:.1f} MB")
                
                total_copied += 1
                
            except Exception as e:
                print(f"  ❌ Error copying {seq_name}: {e}")
    
    print(f"\n✅ Improved sample dataset created!")
    print(f"📊 Summary:")
    print(f"  - Total sequences: {total_copied}")
    print(f"  - Total size: {total_size_mb:.1f} MB")
    
    return target_path

def verify_improved_distribution(improved_samples):
    """Verify the improved class distribution."""
    
    print(f"\n📊 IMPROVED CLASS DISTRIBUTION VERIFICATION")
    print(f"=" * 60)
    
    class_names = ['Car', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Bus', 'Static', 'Other']
    
    # Load original analysis for comparison
    with open('/home/oeoiewt/eTraM/rvt_eTram/dataset_analysis.json', 'r') as f:
        original_results = json.load(f)
    
    for split in ['train', 'val', 'test']:
        if not improved_samples[split]:
            continue
            
        # Original distribution
        original_sequences = list(original_results[split].values())
        original_class_dist = np.sum([s['class_distribution'] for s in original_sequences], axis=0)
        original_total = np.sum(original_class_dist)
        
        # Improved sample distribution
        sample_class_dist = np.sum([s['stats']['class_distribution'] for s in improved_samples[split]], axis=0)
        sample_total = np.sum(sample_class_dist)
        
        print(f'\n{split.upper()} SPLIT VERIFICATION:')
        print(f'Sample total objects: {sample_total:,} ({sample_total/original_total*100:.1f}% of original)')
        print()
        print('Class Name      Original %    Sample %     Difference')
        print('-' * 50)
        
        max_diff = 0
        for i, name in enumerate(class_names):
            orig_pct = original_class_dist[i] / original_total * 100 if original_total > 0 else 0
            sample_pct = sample_class_dist[i] / sample_total * 100 if sample_total > 0 else 0
            diff = abs(sample_pct - orig_pct)
            max_diff = max(max_diff, diff)
            
            status = "✅" if diff <= 3.0 else "⚠️" if diff <= 5.0 else "❌"
            print(f'{name:<12} {orig_pct:>8.1f}%  {sample_pct:>8.1f}%  {sample_pct-orig_pct:>+8.1f}% {status}')
        
        print(f'\nMax difference: {max_diff:.1f}% {"(Good)" if max_diff <= 3.0 else "(Needs improvement)" if max_diff <= 5.0 else "(Poor)"}')

if __name__ == "__main__":
    print("🔄 Starting improved stratified sampling...")
    
    # Perform improved sampling
    improved_samples = improved_stratified_sampling()
    
    # Verify distribution
    verify_improved_distribution(improved_samples)
    
    # Create improved dataset
    create_improved_sample_dataset(improved_samples)
    
    # Save improved samples
    with open('/home/oeoiewt/eTraM/rvt_eTram/improved_samples.json', 'w') as f:
        json.dump(improved_samples, f, indent=2)
    
    print(f"\n✅ Improved sampling complete!")
    print(f"📁 Results saved to: improved_samples.json")
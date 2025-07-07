#!/usr/bin/env python3
"""
Extended sampling for comprehensive long-term training (45 sequences)
"""

import json
import numpy as np
from pathlib import Path
import shutil

def extended_stratified_sampling():
    """Perform extended sampling with 45 sequences for comprehensive training."""
    
    # Load full analysis
    with open('/home/oeoiewt/eTraM/rvt_eTram/dataset_analysis.json', 'r') as f:
        results = json.load(f)
    
    extended_samples = {
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
        
        print(f"\n=== {split.upper()} SPLIT EXTENDED SAMPLING ===")
        print(f"Target class ratios: {[f'{r:.3f}' for r in target_class_ratios]}")
        
        # Extended target sample sizes for comprehensive coverage
        if split == 'train':
            target_size = 30  # Significantly increased for training
        elif split == 'val':
            target_size = 8   # Increased for better validation
        else:  # test
            target_size = 7   # Increased for robust testing
            
        print(f"Target sample size: {target_size} sequences")
        
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
            
            # Additional scoring factors for extended dataset
            small_ratio_score = seq['small_ratio']  # Small object diversity
            object_count_score = min(seq['total_objects'] / 10000, 1.0)  # Normalized object count
            night_bonus = 0.15 if seq['is_night'] else 0.0  # Slight preference for night data
            
            # Class diversity bonus - reward sequences with more non-zero classes
            non_zero_classes = np.count_nonzero(seq_class_dist)
            class_diversity_bonus = (non_zero_classes - 2) * 0.1  # Bonus for 3+ classes
            
            # Size diversity bonus - reward sequences with good mix of small/medium/large
            size_diversity = 1.0 - np.std([seq['size_distribution']['small'], 
                                          seq['size_distribution']['medium'], 
                                          seq['size_distribution']['large']]) / seq_total
            
            # Combined score (lower KL divergence is better, so invert it)
            combined_score = (1.0 / (1.0 + kl_div) + 
                            0.25 * small_ratio_score + 
                            0.15 * object_count_score + 
                            night_bonus +
                            0.1 * class_diversity_bonus +
                            0.1 * size_diversity)
            
            sequence_scores.append({
                'sequence': seq,
                'score': combined_score,
                'kl_div': kl_div,
                'small_ratio': small_ratio_score,
                'class_diversity': non_zero_classes,
                'size_diversity': size_diversity
            })
        
        # Sort by score (descending)
        sequence_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Extended selection strategy with better balance
        selected = []
        day_count = 0
        night_count = 0
        
        # More flexible day/night ratio for extended dataset
        target_night_ratio = 0.4  # 40% night sequences
        max_night = int(target_size * target_night_ratio) + 1
        max_day = target_size - max_night + 2
        
        # First pass: select top scored sequences with balance constraints
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
        
        # Second pass: fill remaining spots with best available
        remaining_slots = target_size - len(selected)
        if remaining_slots > 0:
            for score_info in sequence_scores:
                if len(selected) >= target_size:
                    break
                if score_info not in selected:
                    selected.append(score_info)
        
        # Store selected samples
        for score_info in selected:
            extended_samples[split].append({
                'sequence_name': score_info['sequence']['sequence_name'],
                'stats': score_info['sequence'],
                'selection_score': score_info['score'],
                'kl_divergence': score_info['kl_div'],
                'class_diversity': score_info['class_diversity'],
                'size_diversity': score_info['size_diversity']
            })
        
        # Print detailed selection summary
        selected_class_dist = np.sum([s['stats']['class_distribution'] for s in extended_samples[split]], axis=0)
        selected_total_objects = np.sum(selected_class_dist)
        selected_class_ratios = selected_class_dist / selected_total_objects if selected_total_objects > 0 else np.zeros_like(selected_class_dist)
        
        print(f"Selected {len(extended_samples[split])} sequences:")
        print(f"Total objects: {selected_total_objects:,}")
        print(f"Day/Night split: {day_count} day, {night_count} night")
        print(f"Selected class ratios: {[f'{r:.3f}' for r in selected_class_ratios]}")
        print(f"Class ratio differences: {[f'{s-t:+.3f}' for s, t in zip(selected_class_ratios, target_class_ratios)]}")
        
        # Show top sequences
        for i, sample in enumerate(extended_samples[split]):
            seq_name = sample['sequence_name']
            stats = sample['stats']
            score_info = sample
            day_night = "🌙" if stats['is_night'] else "☀️"
            print(f"  {i+1:2d}. {day_night} {seq_name}: score={score_info['selection_score']:.3f}, "
                  f"objects={stats['total_objects']:,}, small_ratio={stats['small_ratio']:.3f}, "
                  f"classes={score_info['class_diversity']}")
    
    return extended_samples

def create_extended_sample_dataset(extended_samples):
    """Create extended sample dataset with 45 sequences."""
    
    # Remove old sample dataset
    old_target_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_sample')
    if old_target_path.exists():
        shutil.rmtree(old_target_path)
        print(f"🗑️  Removed old sample dataset")
    
    # Create new extended sample dataset
    source_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8')
    target_path = Path('/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_extended_sample')
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Creating extended sample dataset at: {target_path}")
    
    total_copied = 0
    total_size_mb = 0
    
    for split in ['train', 'val', 'test']:
        if not extended_samples[split]:
            continue
            
        print(f"\n📁 Processing {split} split ({len(extended_samples[split])} sequences)...")
        
        for i, sample in enumerate(extended_samples[split]):
            seq_name = sample['sequence_name']
            source_seq_path = source_path / split / seq_name
            target_seq_path = target_path / split / seq_name
            
            if not source_seq_path.exists():
                print(f"  ⚠️  Warning: {source_seq_path} does not exist")
                continue
            
            print(f"  {i+1:2d}. Copying {seq_name}... ", end="")
            
            try:
                target_seq_path.mkdir(exist_ok=True)
                
                sequence_size_mb = 0
                for subdir in ['event_representations_v2', 'labels_v2']:
                    source_subdir = source_seq_path / subdir
                    target_subdir = target_seq_path / subdir
                    
                    if source_subdir.exists():
                        if target_subdir.exists():
                            target_subdir.unlink() if target_subdir.is_symlink() else shutil.rmtree(target_subdir)
                        target_subdir.symlink_to(source_subdir.absolute())
                        
                        size_mb = sum(f.stat().st_size for f in source_subdir.rglob('*') if f.is_file()) / (1024*1024)
                        sequence_size_mb += size_mb
                
                total_size_mb += sequence_size_mb
                total_copied += 1
                print(f"✅ {sequence_size_mb:.1f} MB")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print(f"\n✅ Extended sample dataset created!")
    print(f"📊 Summary:")
    print(f"  - Total sequences: {total_copied}")
    print(f"  - Total size: {total_size_mb:.1f} MB")
    print(f"  - Average per sequence: {total_size_mb/total_copied:.1f} MB")
    
    return target_path

def verify_extended_distribution(extended_samples):
    """Verify the extended class distribution."""
    
    print(f"\n📊 EXTENDED DATASET CLASS DISTRIBUTION VERIFICATION")
    print(f"=" * 65)
    
    class_names = ['Car', 'Truck', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Bus', 'Static', 'Other']
    
    # Load original analysis for comparison
    with open('/home/oeoiewt/eTraM/rvt_eTram/dataset_analysis.json', 'r') as f:
        original_results = json.load(f)
    
    grand_total_original = 0
    grand_total_sample = 0
    
    for split in ['train', 'val', 'test']:
        if not extended_samples[split]:
            continue
            
        # Original distribution
        original_sequences = list(original_results[split].values())
        original_class_dist = np.sum([s['class_distribution'] for s in original_sequences], axis=0)
        original_total = np.sum(original_class_dist)
        grand_total_original += original_total
        
        # Extended sample distribution
        sample_class_dist = np.sum([s['stats']['class_distribution'] for s in extended_samples[split]], axis=0)
        sample_total = np.sum(sample_class_dist)
        grand_total_sample += sample_total
        
        print(f'\n{split.upper()} SPLIT VERIFICATION:')
        print(f'Sequences: {len(extended_samples[split])} / {len(original_sequences)} '
              f'({len(extended_samples[split])/len(original_sequences)*100:.1f}%)')
        print(f'Objects: {sample_total:,} / {original_total:,} '
              f'({sample_total/original_total*100:.1f}%)')
        print()
        print('Class Name      Original %    Sample %     Difference')
        print('-' * 55)
        
        max_diff = 0
        for i, name in enumerate(class_names):
            orig_pct = original_class_dist[i] / original_total * 100 if original_total > 0 else 0
            sample_pct = sample_class_dist[i] / sample_total * 100 if sample_total > 0 else 0
            diff = abs(sample_pct - orig_pct)
            max_diff = max(max_diff, diff)
            
            status = "✅" if diff <= 2.0 else "⚠️" if diff <= 4.0 else "❌"
            print(f'{name:<12} {orig_pct:>8.1f}%  {sample_pct:>8.1f}%  {sample_pct-orig_pct:>+8.1f}% {status}')
        
        print(f'\nMax difference: {max_diff:.1f}% {"(Excellent)" if max_diff <= 2.0 else "(Good)" if max_diff <= 4.0 else "(Needs improvement)"}')
    
    print(f'\n🎯 OVERALL DATASET SUMMARY:')
    print(f'Total sample coverage: {grand_total_sample:,} / {grand_total_original:,} objects '
          f'({grand_total_sample/grand_total_original*100:.1f}%)')

def create_extended_config():
    """Create configuration for extended dataset."""
    config_content = """defaults:
  - gen4

name: gen4_extended_sample
path: /home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_extended_sample

# Extended sample configuration
extended_sample: true
sample_size: 45
coverage_target: 0.15  # ~15% of full dataset objects

# Training optimizations for extended dataset
batch_optimization: true
sequence_balancing: true
"""
    
    config_path = Path('/home/oeoiewt/eTraM/rvt_eTram/config/dataset/gen4_extended_sample.yaml')
    config_path.write_text(config_content)
    print(f"📄 Created dataset config: {config_path}")
    return config_path

if __name__ == "__main__":
    print("🚀 Starting extended stratified sampling for long-term training...")
    print("Target: 45 sequences (30 train + 8 val + 7 test)")
    
    # Perform extended sampling
    extended_samples = extended_stratified_sampling()
    
    # Verify distribution
    verify_extended_distribution(extended_samples)
    
    # Create extended dataset
    target_path = create_extended_sample_dataset(extended_samples)
    
    # Create configuration
    config_path = create_extended_config()
    
    # Save extended samples
    with open('/home/oeoiewt/eTraM/rvt_eTram/extended_samples.json', 'w') as f:
        json.dump(extended_samples, f, indent=2)
    
    print(f"\n✅ Extended sampling complete!")
    print(f"📁 Dataset: {target_path}")
    print(f"📄 Config: {config_path}")
    print(f"📊 Results: extended_samples.json")
    print(f"\n🎯 Ready for 12-24 hour comprehensive training!")
#!/usr/bin/env python3
"""
Dataset Size Analysis for Size-aware Loss Optimization
=====================================================

This script analyzes the bounding box size distribution in the eTraM dataset
to optimize size-aware loss thresholds for small object detection.

Key objectives:
1. Analyze size distribution of each object class
2. Identify optimal thresholds for small/medium/large objects
3. Calculate class-balanced thresholds for size-aware loss
4. Generate recommendations for threshold tuning

Usage:
    python utils/dataset_size_analysis.py --dataset_path /path/to/dataset --output_path analysis_results.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Class mapping for eTraM dataset
CLASS_NAMES = {
    0: "Car",
    1: "Truck", 
    2: "Motorcycle",
    3: "Bicycle",
    4: "Pedestrian",
    5: "Bus",
    6: "Static",
    7: "Other"
}

# Small object classes (target for improvement)
SMALL_OBJECT_CLASSES = [2, 3, 4]  # Motorcycle, Bicycle, Pedestrian
MEDIUM_OBJECT_CLASSES = [0, 1, 5]  # Car, Truck, Bus  
LARGE_OBJECT_CLASSES = [6, 7]     # Static, Other

class DatasetSizeAnalyzer:
    """Analyzes bounding box size distribution in eTraM dataset"""
    
    def __init__(self, dataset_path: str, output_path: str = "dataset_analysis.json"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.bbox_data = defaultdict(list)  # class_id -> list of (width, height, area)
        self.class_counts = defaultdict(int)
        self.total_objects = 0
        
    def load_annotations(self):
        """Load all bounding box annotations from the dataset"""
        print(f"Loading annotations from {self.dataset_path}")
        
        # Process train, val, test sets
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                print(f"Warning: {split} split not found, skipping...")
                continue
                
            print(f"Processing {split} split...")
            self._process_split(split_path)
            
        print(f"Total objects loaded: {self.total_objects}")
        print(f"Classes found: {list(self.class_counts.keys())}")
        
    def _process_split(self, split_path: Path):
        """Process a single split (train/val/test)"""
        
        for sequence_dir in split_path.iterdir():
            if not sequence_dir.is_dir():
                continue
                
            labels_dir = sequence_dir / 'labels_v2'
            if not labels_dir.exists():
                continue
                
            # Process all label files in sequence
            for label_file in labels_dir.glob('*.txt'):
                self._process_label_file(label_file)
                
    def _process_label_file(self, label_file: Path):
        """Process a single label file"""
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                # Assuming input image size of 640x360 (from training config)
                img_width, img_height = 640, 360
                pixel_width = width * img_width
                pixel_height = height * img_height
                area = pixel_width * pixel_height
                
                # Store bbox data
                self.bbox_data[class_id].append({
                    'width': pixel_width,
                    'height': pixel_height,
                    'area': area,
                    'normalized_width': width,
                    'normalized_height': height
                })
                
                self.class_counts[class_id] += 1
                self.total_objects += 1
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            
    def analyze_size_distribution(self) -> Dict:
        """Analyze size distribution across all classes"""
        print("\\nAnalyzing size distribution...")
        
        analysis = {
            'total_objects': self.total_objects,
            'class_counts': dict(self.class_counts),
            'class_statistics': {},
            'size_categories': {},
            'threshold_recommendations': {}
        }
        
        # Analyze each class
        for class_id, bbox_list in self.bbox_data.items():
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            areas = [bbox['area'] for bbox in bbox_list]
            widths = [bbox['width'] for bbox in bbox_list]
            heights = [bbox['height'] for bbox in bbox_list]
            
            stats = {
                'count': len(bbox_list),
                'class_name': class_name,
                'area_stats': {
                    'mean': np.mean(areas),
                    'std': np.std(areas),
                    'min': np.min(areas),
                    'max': np.max(areas),
                    'median': np.median(areas),
                    'q25': np.percentile(areas, 25),
                    'q75': np.percentile(areas, 75),
                    'q90': np.percentile(areas, 90),
                    'q95': np.percentile(areas, 95)
                },
                'width_stats': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths),
                    'median': np.median(widths)
                },
                'height_stats': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights),
                    'median': np.median(heights)
                }
            }
            
            analysis['class_statistics'][class_id] = stats
            
        # Analyze size categories
        small_areas = []
        medium_areas = []
        large_areas = []
        
        for class_id, bbox_list in self.bbox_data.items():
            areas = [bbox['area'] for bbox in bbox_list]
            
            if class_id in SMALL_OBJECT_CLASSES:
                small_areas.extend(areas)
            elif class_id in MEDIUM_OBJECT_CLASSES:
                medium_areas.extend(areas)
            elif class_id in LARGE_OBJECT_CLASSES:
                large_areas.extend(areas)
                
        analysis['size_categories'] = {
            'small_objects': {
                'classes': [CLASS_NAMES[i] for i in SMALL_OBJECT_CLASSES],
                'count': len(small_areas),
                'area_stats': self._calculate_stats(small_areas) if small_areas else {}
            },
            'medium_objects': {
                'classes': [CLASS_NAMES[i] for i in MEDIUM_OBJECT_CLASSES],
                'count': len(medium_areas),
                'area_stats': self._calculate_stats(medium_areas) if medium_areas else {}
            },
            'large_objects': {
                'classes': [CLASS_NAMES[i] for i in LARGE_OBJECT_CLASSES],
                'count': len(large_areas),
                'area_stats': self._calculate_stats(large_areas) if large_areas else {}
            }
        }
        
        # Generate threshold recommendations
        analysis['threshold_recommendations'] = self._generate_threshold_recommendations(
            small_areas, medium_areas, large_areas
        )
        
        return analysis
        
    def _calculate_stats(self, values: List[float]) -> Dict:
        """Calculate statistics for a list of values"""
        if not values:
            return {}
            
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'q90': np.percentile(values, 90),
            'q95': np.percentile(values, 95),
            'q99': np.percentile(values, 99)
        }
        
    def _generate_threshold_recommendations(self, small_areas: List[float], 
                                          medium_areas: List[float], 
                                          large_areas: List[float]) -> Dict:
        """Generate threshold recommendations for size-aware loss"""
        
        recommendations = {}
        
        if small_areas:
            # Small threshold: Include most small objects (90th percentile)
            small_threshold = np.percentile(small_areas, 90)
            recommendations['small_threshold'] = {
                'value': int(small_threshold),
                'rationale': f"90th percentile of small objects ({len(small_areas)} samples)",
                'coverage': "90% of small objects will be weighted"
            }
            
        if medium_areas:
            # Medium threshold: Boundary between medium and large (75th percentile)
            medium_threshold = np.percentile(medium_areas, 75)
            recommendations['medium_threshold'] = {
                'value': int(medium_threshold),
                'rationale': f"75th percentile of medium objects ({len(medium_areas)} samples)",
                'coverage': "75% of medium objects will receive intermediate weighting"
            }
            
        # Alternative approach: Use overall distribution
        all_areas = small_areas + medium_areas + large_areas
        if all_areas:
            recommendations['alternative_thresholds'] = {
                'small_threshold_p30': {
                    'value': int(np.percentile(all_areas, 30)),
                    'rationale': "30th percentile of all objects (emphasizes truly small objects)"
                },
                'medium_threshold_p70': {
                    'value': int(np.percentile(all_areas, 70)),
                    'rationale': "70th percentile of all objects (balanced approach)"
                },
                'small_threshold_p25': {
                    'value': int(np.percentile(all_areas, 25)),
                    'rationale': "25th percentile of all objects (conservative approach)"
                }
            }
            
        # Current baseline comparison
        recommendations['current_baseline'] = {
            'small_threshold': 1024,
            'medium_threshold': 9216,
            'rationale': "Current settings (32x32 and 96x96 pixels)"
        }
        
        # Weight suggestions
        if small_areas and medium_areas:
            count_ratio = len(medium_areas) / len(small_areas) if small_areas else 1.0
            recommended_weight = min(max(count_ratio * 1.5, 2.0), 5.0)
            
            recommendations['weight_suggestions'] = {
                'size_aware_weight': recommended_weight,
                'rationale': f"Based on class imbalance ratio: {count_ratio:.2f}",
                'weight_type': 'exponential'
            }
            
        return recommendations
        
    def generate_visualizations(self, analysis: Dict, output_dir: str = "analysis_plots"):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Class distribution bar plot
        self._plot_class_distribution(analysis, output_dir)
        
        # 2. Size distribution histograms
        self._plot_size_distributions(analysis, output_dir)
        
        # 3. Box plots by class
        self._plot_size_boxplots(analysis, output_dir)
        
        # 4. Threshold visualization
        self._plot_threshold_analysis(analysis, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
        
    def _plot_class_distribution(self, analysis: Dict, output_dir: Path):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count distribution
        classes = []
        counts = []
        for class_id, count in analysis['class_counts'].items():
            classes.append(CLASS_NAMES.get(class_id, f"Class_{class_id}"))
            counts.append(count)
            
        ax1.bar(classes, counts)
        ax1.set_title('Object Count by Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Percentage distribution
        total = sum(counts)
        percentages = [c/total*100 for c in counts]
        ax2.bar(classes, percentages)
        ax2.set_title('Object Percentage by Class')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_size_distributions(self, analysis: Dict, output_dir: Path):
        """Plot size distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect all areas by size category
        small_areas = []
        medium_areas = []
        large_areas = []
        
        for class_id, bbox_list in self.bbox_data.items():
            areas = [bbox['area'] for bbox in bbox_list]
            
            if class_id in SMALL_OBJECT_CLASSES:
                small_areas.extend(areas)
            elif class_id in MEDIUM_OBJECT_CLASSES:
                medium_areas.extend(areas)
            elif class_id in LARGE_OBJECT_CLASSES:
                large_areas.extend(areas)
                
        # Plot histograms
        if small_areas:
            axes[0, 0].hist(small_areas, bins=50, alpha=0.7, color='red')
            axes[0, 0].set_title('Small Objects (Motorcycle, Bicycle, Pedestrian)')
            axes[0, 0].set_xlabel('Area (pixels²)')
            axes[0, 0].set_ylabel('Frequency')
            
        if medium_areas:
            axes[0, 1].hist(medium_areas, bins=50, alpha=0.7, color='blue')
            axes[0, 1].set_title('Medium Objects (Car, Truck, Bus)')
            axes[0, 1].set_xlabel('Area (pixels²)')
            axes[0, 1].set_ylabel('Frequency')
            
        if large_areas:
            axes[1, 0].hist(large_areas, bins=50, alpha=0.7, color='green')
            axes[1, 0].set_title('Large Objects (Static, Other)')
            axes[1, 0].set_xlabel('Area (pixels²)')
            axes[1, 0].set_ylabel('Frequency')
            
        # Combined distribution
        all_areas = small_areas + medium_areas + large_areas
        if all_areas:
            axes[1, 1].hist([small_areas, medium_areas, large_areas], 
                          bins=50, alpha=0.7, 
                          label=['Small', 'Medium', 'Large'],
                          color=['red', 'blue', 'green'])
            axes[1, 1].set_title('Combined Size Distribution')
            axes[1, 1].set_xlabel('Area (pixels²)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'size_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_size_boxplots(self, analysis: Dict, output_dir: Path):
        """Plot size boxplots by class"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for class_id in sorted(self.bbox_data.keys()):
            areas = [bbox['area'] for bbox in self.bbox_data[class_id]]
            data.append(areas)
            labels.append(CLASS_NAMES.get(class_id, f"Class_{class_id}"))
            
        ax.boxplot(data, labels=labels)
        ax.set_title('Bounding Box Area Distribution by Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Area (pixels²)')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(output_dir / 'size_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_threshold_analysis(self, analysis: Dict, output_dir: Path):
        """Plot threshold analysis"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all areas
        all_areas = []
        for bbox_list in self.bbox_data.values():
            all_areas.extend([bbox['area'] for bbox in bbox_list])
            
        # Plot histogram
        ax.hist(all_areas, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add threshold lines
        recommendations = analysis['threshold_recommendations']
        
        # Current baseline
        current_small = recommendations['current_baseline']['small_threshold']
        current_medium = recommendations['current_baseline']['medium_threshold']
        ax.axvline(current_small, color='red', linestyle='--', linewidth=2, 
                  label=f'Current Small Threshold ({current_small})')
        ax.axvline(current_medium, color='orange', linestyle='--', linewidth=2,
                  label=f'Current Medium Threshold ({current_medium})')
        
        # Recommended thresholds
        if 'small_threshold' in recommendations:
            rec_small = recommendations['small_threshold']['value']
            ax.axvline(rec_small, color='green', linestyle='-', linewidth=2,
                      label=f'Recommended Small Threshold ({rec_small})')
                      
        if 'medium_threshold' in recommendations:
            rec_medium = recommendations['medium_threshold']['value']
            ax.axvline(rec_medium, color='blue', linestyle='-', linewidth=2,
                      label=f'Recommended Medium Threshold ({rec_medium})')
        
        ax.set_title('Threshold Analysis on Overall Size Distribution')
        ax.set_xlabel('Area (pixels²)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim(0, np.percentile(all_areas, 95))  # Focus on 95th percentile
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_analysis(self, analysis: Dict):
        """Save analysis results to JSON file"""
        with open(self.output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {self.output_path}")
        
    def print_summary(self, analysis: Dict):
        """Print summary of analysis"""
        print("\\n" + "="*60)
        print("DATASET SIZE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total objects: {analysis['total_objects']}")
        print("\\nClass distribution:")
        for class_id, count in analysis['class_counts'].items():
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            percentage = count / analysis['total_objects'] * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
        print("\\nSize category analysis:")
        for category, info in analysis['size_categories'].items():
            print(f"\\n{category.upper()}:")
            print(f"  Classes: {', '.join(info['classes'])}")
            print(f"  Count: {info['count']}")
            if info['area_stats']:
                stats = info['area_stats']
                print(f"  Area range: {stats['min']:.0f} - {stats['max']:.0f} pixels²")
                print(f"  Mean area: {stats['mean']:.0f} pixels²")
                print(f"  Median area: {stats['median']:.0f} pixels²")
                
        print("\\nTHRESHOLD RECOMMENDATIONS:")
        print("="*40)
        
        rec = analysis['threshold_recommendations']
        
        print("\\nCurrent baseline:")
        current = rec['current_baseline']
        print(f"  Small threshold: {current['small_threshold']}")
        print(f"  Medium threshold: {current['medium_threshold']}")
        
        print("\\nRecommended thresholds:")
        if 'small_threshold' in rec:
            small = rec['small_threshold']
            print(f"  Small threshold: {small['value']}")
            print(f"    Rationale: {small['rationale']}")
            print(f"    Coverage: {small['coverage']}")
            
        if 'medium_threshold' in rec:
            medium = rec['medium_threshold']
            print(f"  Medium threshold: {medium['value']}")
            print(f"    Rationale: {medium['rationale']}")
            print(f"    Coverage: {medium['coverage']}")
            
        if 'weight_suggestions' in rec:
            weight = rec['weight_suggestions']
            print(f"\\nWeight suggestions:")
            print(f"  Size-aware weight: {weight['size_aware_weight']:.1f}")
            print(f"  Rationale: {weight['rationale']}")
            print(f"  Weight type: {weight['weight_type']}")
            
        print("\\nAlternative thresholds:")
        if 'alternative_thresholds' in rec:
            alt = rec['alternative_thresholds']
            for name, info in alt.items():
                print(f"  {name}: {info['value']} ({info['rationale']})")

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset size distribution')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample',
                       help='Path to dataset')
    parser.add_argument('--output_path', type=str, 
                       default='dataset_size_analysis.json',
                       help='Output path for analysis results')
    parser.add_argument('--plot_dir', type=str,
                       default='analysis_plots',
                       help='Directory for visualization plots')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DatasetSizeAnalyzer(args.dataset_path, args.output_path)
    
    # Load data and analyze
    analyzer.load_annotations()
    analysis = analyzer.analyze_size_distribution()
    
    # Generate visualizations
    if not args.no_plots:
        analyzer.generate_visualizations(analysis, args.plot_dir)
    
    # Save and print results
    analyzer.save_analysis(analysis)
    analyzer.print_summary(analysis)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Experiment logging system for detailed experiment tracking and comparison
Saves comprehensive experiment results in JSON format with git integration
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Logger for saving detailed experiment results with git integration."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # eTraM 8-class system mapping
        self.class_mapping = {
            0: "Car",           # Large object
            1: "Truck",         # Large object  
            2: "Motorcycle",    # Small object ⭐
            3: "Bicycle",       # Small object ⭐
            4: "Pedestrian",    # Very small object ⭐⭐
            5: "Bus",           # Large object
            6: "Static",        # Static object
            7: "Other"          # Other
        }
        
    def save_experiment_results(self, 
                              experiment_id: str,
                              model_modifications: Dict[str, Any],
                              training_config: Dict[str, Any],
                              metrics_data: Dict[str, Any],
                              additional_info: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save comprehensive experiment results to JSON file.
        
        Args:
            experiment_id: Unique experiment identifier
            model_modifications: Details of model architecture changes
            training_config: Training configuration parameters
            metrics_data: Evaluation metrics from detailed_metrics.py
            additional_info: Optional additional experiment information
            
        Returns:
            Path to saved JSON file
        """
        
        # Get git information
        git_info = self._get_git_info()
        
        # Create comprehensive experiment record
        experiment_record = {
            "experiment_metadata": {
                "experiment_date": datetime.now().isoformat(),
                "experiment_id": experiment_id,
                "git_commit": git_info["commit_hash"],
                "git_branch": git_info["branch"],
                "git_status": git_info["status"],
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            
            "model_modifications": self._format_model_modifications(model_modifications),
            
            "training_config": training_config,
            
            "evaluation_results": {
                "overall_metrics": metrics_data.get("overall_metrics", {}),
                "class_metrics": self._format_class_metrics(metrics_data.get("class_metrics", {})),
                "small_object_analysis": metrics_data.get("small_object_analysis", {}),
                "evaluation_summary": metrics_data.get("evaluation_summary", {})
            },
            
            "performance_highlights": self._extract_performance_highlights(metrics_data),
            
            "additional_info": additional_info or {}
        }
        
        # Save to JSON file
        json_file_path = self.experiment_dir / f"{experiment_id}_results.json"
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=2, ensure_ascii=False)
        
        # Also save a latest results link
        latest_file_path = self.experiment_dir / "latest_results.json"
        with open(latest_file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Experiment results saved: {json_file_path}")
        print(f"📊 Latest results updated: {latest_file_path}")
        
        return json_file_path
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git repository information."""
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            # Get git status
            status_output = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            status = "clean" if not status_output else "modified"
            
            return {
                "commit_hash": commit_hash[:8],  # Short hash
                "branch": branch,
                "status": status
            }
            
        except subprocess.CalledProcessError:
            return {
                "commit_hash": "unknown",
                "branch": "unknown", 
                "status": "unknown"
            }
    
    def _format_model_modifications(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Format model modifications for consistent logging."""
        
        # Ensure all required fields are present
        formatted = {
            "architecture_changes": modifications.get("architecture_changes", []),
            "config_changes": modifications.get("config_changes", []),
            "baseline_vs_modified": modifications.get("baseline_vs_modified", ""),
            "key_features": modifications.get("key_features", []),
            "implementation_notes": modifications.get("implementation_notes", [])
        }
        
        return formatted
    
    def _format_class_metrics(self, class_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format class metrics with proper naming and organization."""
        
        formatted_metrics = {}
        
        for class_key, metrics in class_metrics.items():
            # Extract class ID and name
            if "_" in class_key:
                class_id_str, class_name = class_key.split("_", 1)
                try:
                    class_id = int(class_id_str)
                except ValueError:
                    class_id = -1
            else:
                class_id = -1
                class_name = class_key
            
            # Determine object size category
            size_category = self._get_size_category(class_id)
            
            formatted_metrics[class_key] = {
                "class_info": {
                    "id": class_id,
                    "name": class_name,
                    "size_category": size_category
                },
                "metrics": {
                    "mAP": round(metrics.get("mAP", 0.0), 4),
                    "AP50": round(metrics.get("AP50", 0.0), 4),
                    "AP75": round(metrics.get("AP75", 0.0), 4),
                    "AP95": round(metrics.get("AP95", 0.0), 4)
                },
                "detection_stats": {
                    "count": metrics.get("count", 0),
                    "detection_rate": self._calculate_detection_rate(metrics)
                }
            }
        
        return formatted_metrics
    
    def _get_size_category(self, class_id: int) -> str:
        """Determine size category of object class."""
        
        small_objects = [2, 3, 4]  # Motorcycle, Bicycle, Pedestrian
        large_objects = [0, 1, 5]  # Car, Truck, Bus
        static_objects = [6, 7]    # Static, Other
        
        if class_id in small_objects:
            return "small"
        elif class_id in large_objects:
            return "large"
        elif class_id in static_objects:
            return "static"
        else:
            return "unknown"
    
    def _calculate_detection_rate(self, metrics: Dict[str, Any]) -> str:
        """Calculate detection rate description based on metrics."""
        
        map_value = metrics.get("mAP", 0.0)
        count = metrics.get("count", 0)
        
        if count == 0:
            return "no_detections"
        elif map_value > 0.3:
            return "good"
        elif map_value > 0.1:
            return "moderate"
        elif map_value > 0.0:
            return "poor"
        else:
            return "failed"
    
    def _extract_performance_highlights(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance highlights from metrics."""
        
        overall = metrics_data.get("overall_metrics", {})
        class_metrics = metrics_data.get("class_metrics", {})
        small_obj = metrics_data.get("small_object_analysis", {})
        
        # Find best and worst performing classes
        class_performances = []
        for class_key, metrics in class_metrics.items():
            if isinstance(metrics, dict) and "mAP" in metrics:
                class_performances.append((class_key, metrics["mAP"]))
        
        class_performances.sort(key=lambda x: x[1], reverse=True)
        
        highlights = {
            "overall_performance": {
                "mAP": round(overall.get("mAP", 0.0), 4),
                "AP50": round(overall.get("AP50", 0.0), 4),
                "best_metric": "AP50" if overall.get("AP50", 0) > overall.get("mAP", 0) else "mAP"
            },
            
            "class_performance": {
                "best_class": class_performances[0] if class_performances else ("None", 0.0),
                "worst_class": class_performances[-1] if class_performances else ("None", 0.0),
                "active_classes": len([p for p in class_performances if p[1] > 0])
            },
            
            "small_object_performance": {
                "avg_small_mAP": round(small_obj.get("avg_small_mAP", 0.0), 4),
                "small_object_count": small_obj.get("small_object_count", 0),
                "small_classes_detected": len([m for m in small_obj.get("individual_maps", {}).values() if m > 0])
            }
        }
        
        return highlights
    
    def create_experiment_comparison(self, experiment_ids: list) -> Dict[str, Any]:
        """Create a comparison between multiple experiments."""
        
        comparison_data = {
            "comparison_metadata": {
                "created_date": datetime.now().isoformat(),
                "experiments_compared": experiment_ids,
                "comparison_id": f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "experiments": {},
            "performance_comparison": {},
            "improvement_analysis": {}
        }
        
        # Load experiment data
        for exp_id in experiment_ids:
            exp_file = self.experiment_dir / f"{exp_id}_results.json"
            if exp_file.exists():
                with open(exp_file, 'r') as f:
                    comparison_data["experiments"][exp_id] = json.load(f)
        
        # Analyze improvements (if we have baseline)
        if len(comparison_data["experiments"]) >= 2:
            comparison_data["improvement_analysis"] = self._analyze_improvements(
                comparison_data["experiments"]
            )
        
        # Save comparison
        comparison_file = self.experiment_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"📊 Experiment comparison saved: {comparison_file}")
        return comparison_data
    
    def _analyze_improvements(self, experiments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze improvements between experiments."""
        
        # Find baseline (assume first experiment or one with "baseline" in name)
        baseline_key = None
        for key in experiments.keys():
            if "baseline" in key.lower() or "3scale" in key.lower():
                baseline_key = key
                break
        
        if baseline_key is None:
            baseline_key = list(experiments.keys())[0]
        
        baseline = experiments[baseline_key]
        improvements = {}
        
        for exp_key, exp_data in experiments.items():
            if exp_key == baseline_key:
                continue
                
            # Compare overall metrics
            baseline_map = baseline.get("evaluation_results", {}).get("overall_metrics", {}).get("mAP", 0)
            current_map = exp_data.get("evaluation_results", {}).get("overall_metrics", {}).get("mAP", 0)
            
            improvements[exp_key] = {
                "overall_improvement": {
                    "mAP_change": round(current_map - baseline_map, 4),
                    "mAP_percent_change": round(((current_map - baseline_map) / baseline_map * 100) if baseline_map > 0 else 0, 2)
                },
                "small_object_improvement": self._compare_small_objects(baseline, exp_data)
            }
        
        return improvements
    
    def _compare_small_objects(self, baseline: Dict, current: Dict) -> Dict[str, Any]:
        """Compare small object performance between experiments."""
        
        baseline_small = baseline.get("evaluation_results", {}).get("small_object_analysis", {})
        current_small = current.get("evaluation_results", {}).get("small_object_analysis", {})
        
        baseline_avg = baseline_small.get("avg_small_mAP", 0)
        current_avg = current_small.get("avg_small_mAP", 0)
        
        return {
            "avg_small_mAP_change": round(current_avg - baseline_avg, 4),
            "avg_small_mAP_percent_change": round(((current_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0, 2),
            "baseline_avg": round(baseline_avg, 4),
            "current_avg": round(current_avg, 4)
        }
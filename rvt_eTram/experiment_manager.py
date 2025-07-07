#!/usr/bin/env python3
"""
Experiment Management System for 4-scale FPN Long-run Training
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import psutil
import subprocess

class ExperimentManager:
    def __init__(self, experiment_name="4scale_longrun"):
        self.base_dir = Path("/home/oeoiewt/eTraM/rvt_eTram")
        self.experiments_dir = self.base_dir / "experiments"
        
        # Create experiment ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.experiment_dir = self.experiments_dir / self.experiment_id
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize experiment tracking
        self.experiment_log = {
            "experiment_id": self.experiment_id,
            "experiment_start": datetime.now().isoformat(),
            "current_phase": None,
            "current_step": 0,
            "status": "initializing",
            "milestones": [],
            "alerts": [],
            "performance_tracking": {}
        }
        
        self.experiment_plan = None
        
    def setup_directories(self):
        """Create experiment directory structure."""
        dirs = [
            "checkpoints",
            "confusion_matrices", 
            "logs",
            "results"
        ]
        
        for dir_name in dirs:
            (self.experiment_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Experiment directory created: {self.experiment_dir}")
    
    def create_experiment_plan(self, config):
        """Create detailed experiment plan."""
        self.experiment_plan = {
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "objective": "4-scale FPN small object detection performance validation",
            "hypothesis": "P1 features (stride 4) will improve small object AP by 15-25%",
            "dataset": config.get("dataset", {}),
            "model": {
                "architecture": "RVT + 4-scale FPN",
                "scales": [4, 8, 16, 32],
                "backbone": "MaxViT-RNN",
                "modifications": "Added P1 feature processing"
            },
            "training": config.get("training", {}),
            "success_criteria": {
                "convergence": "validation loss < 10",
                "class_coverage": "all 8 classes detected",
                "small_object_improvement": "AP_small > baseline + 10%"
            },
            "monitoring": {
                "validation_interval": config.get("training", {}).get("validation_interval", 2000),
                "checkpoint_interval": config.get("training", {}).get("checkpoint_interval", 5000),
                "confusion_matrix_interval": 2000
            }
        }
        
        # Save experiment plan
        plan_file = self.experiment_dir / "experiment_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(self.experiment_plan, f, indent=2)
        
        print(f"📋 Experiment plan saved: {plan_file}")
        return self.experiment_plan
    
    def log_milestone(self, event, details=None, step=None, metrics=None):
        """Log important milestones during training."""
        milestone = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        
        if step is not None:
            milestone["step"] = step
        if details:
            milestone["details"] = details
        if metrics:
            milestone["metrics"] = metrics
            
        self.experiment_log["milestones"].append(milestone)
        
        # Update current status
        if step is not None:
            self.experiment_log["current_step"] = step
            
        # Save updated log
        self.save_experiment_log()
        
        # Also append to monitor file for real-time viewing
        self.append_to_monitor(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}")
        if details:
            self.append_to_monitor(f"  Details: {details}")
        if metrics:
            self.append_to_monitor(f"  Metrics: {metrics}")
    
    def log_alert(self, alert_type, message):
        """Log alerts and warnings."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        
        self.experiment_log["alerts"].append(alert)
        self.save_experiment_log()
        
        # Add to monitor file
        self.append_to_monitor(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {alert_type.upper()}: {message}")
    
    def update_performance_tracking(self, metrics):
        """Update performance tracking metrics."""
        self.experiment_log["performance_tracking"].update(metrics)
        self.save_experiment_log()
    
    def update_status(self, status, phase=None):
        """Update experiment status."""
        self.experiment_log["status"] = status
        if phase:
            self.experiment_log["current_phase"] = phase
        self.save_experiment_log()
    
    def save_experiment_log(self):
        """Save experiment log to file."""
        log_file = self.experiment_dir / "experiment_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
    
    def append_to_monitor(self, message):
        """Append message to real-time monitor file."""
        monitor_file = self.experiment_dir / "training_monitor.txt"
        with open(monitor_file, 'a') as f:
            f.write(f"{message}\n")
    
    def get_system_info(self):
        """Get current system information."""
        try:
            # GPU info
            gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                     '--format=csv,nounits,noheader'], 
                                    capture_output=True, text=True)
            
            if gpu_info.returncode == 0:
                gpu_lines = gpu_info.stdout.strip().split('\n')
                gpu_data = gpu_lines[0].split(', ') if gpu_lines else ['0', '0', '0']
                gpu_used, gpu_total, gpu_util = gpu_data
                
                system_info = {
                    "gpu_memory_used_mb": int(gpu_used),
                    "gpu_memory_total_mb": int(gpu_total),
                    "gpu_utilization_percent": int(gpu_util),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/home').percent
                }
            else:
                system_info = {
                    "gpu_memory_used_mb": "N/A",
                    "gpu_memory_total_mb": "N/A", 
                    "gpu_utilization_percent": "N/A",
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/home').percent
                }
                
            return system_info
            
        except Exception as e:
            return {"error": str(e)}
    
    def start_system_monitoring(self):
        """Start system monitoring in background."""
        system_log_file = self.experiment_dir / "logs" / "system_monitor.log"
        
        def monitor_loop():
            while True:
                system_info = self.get_system_info()
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "system_info": system_info
                }
                
                with open(system_log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                time.sleep(60)  # Monitor every minute
        
        # This would typically run in a separate thread/process
        # For now, just log the initial state
        initial_info = self.get_system_info()
        with open(system_log_file, 'w') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "monitoring_started",
                "system_info": initial_info
            }) + '\n')
        
        return initial_info
    
    def finalize_experiment(self, final_metrics=None):
        """Finalize experiment and create summary."""
        self.experiment_log["experiment_end"] = datetime.now().isoformat()
        self.experiment_log["status"] = "completed"
        
        if final_metrics:
            self.experiment_log["final_metrics"] = final_metrics
        
        self.save_experiment_log()
        
        # Create summary report
        summary = self.create_summary_report()
        summary_file = self.experiment_dir / "results" / "summary_report.md"
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Experiment completed: {self.experiment_id}")
        print(f"📁 Results saved in: {self.experiment_dir}")
        
        return summary_file
    
    def create_summary_report(self):
        """Create markdown summary report."""
        duration = "N/A"
        if "experiment_end" in self.experiment_log:
            start = datetime.fromisoformat(self.experiment_log["experiment_start"])
            end = datetime.fromisoformat(self.experiment_log["experiment_end"])
            duration = str(end - start)
        
        summary = f"""# Experiment Summary: {self.experiment_id}

## Overview
- **Objective**: 4-scale FPN small object detection validation
- **Start Time**: {self.experiment_log["experiment_start"]}
- **Duration**: {duration}
- **Status**: {self.experiment_log["status"]}

## Key Milestones
"""
        
        for milestone in self.experiment_log["milestones"][-10:]:  # Last 10 milestones
            summary += f"- **{milestone['event']}** ({milestone['timestamp']})\n"
            if 'metrics' in milestone:
                summary += f"  - Metrics: {milestone['metrics']}\n"
        
        summary += f"""

## Alerts ({len(self.experiment_log["alerts"])})
"""
        
        for alert in self.experiment_log["alerts"][-5:]:  # Last 5 alerts
            summary += f"- **{alert['type']}**: {alert['message']} ({alert['timestamp']})\n"
        
        summary += f"""

## Performance Summary
- **Current Step**: {self.experiment_log["current_step"]}
- **Current Phase**: {self.experiment_log.get("current_phase", "N/A")}

## Files Generated
- Checkpoints: `{self.experiment_dir}/checkpoints/`
- Confusion Matrices: `{self.experiment_dir}/confusion_matrices/`
- Logs: `{self.experiment_dir}/logs/`
- Real-time Monitor: `{self.experiment_dir}/training_monitor.txt`

## Access Commands
```bash
# Monitor real-time progress
tail -f {self.experiment_dir}/training_monitor.txt

# Check experiment status
cat {self.experiment_dir}/experiment_log.json

# View latest confusion matrix
ls -la {self.experiment_dir}/confusion_matrices/

# Check system logs
tail -f {self.experiment_dir}/logs/system_monitor.log
```
"""
        
        return summary

def create_experiment_config(dataset_size="extended", training_steps=100000):
    """Create experiment configuration."""
    config = {
        "dataset": {
            "name": "gen4_extended_sample",
            "sequences": 45 if dataset_size == "extended" else 16,
            "target_objects": 350000 if dataset_size == "extended" else 116000,
            "class_balance": "enforced",
            "small_object_ratio": 0.28
        },
        "training": {
            "total_steps": training_steps,
            "phases": [
                {"phase": "A", "steps": 20000, "focus": "basic_convergence"},
                {"phase": "B", "steps": 50000, "focus": "optimization"}, 
                {"phase": "C", "steps": training_steps, "focus": "final_tuning"}
            ],
            "validation_interval": 2000,
            "checkpoint_interval": 5000,
            "batch_size": {"train": 6, "eval": 2},
            "learning_rate": 3.5e-5
        }
    }
    return config

if __name__ == "__main__":
    # Example usage
    print("🚀 Setting up experiment management system...")
    
    # Create experiment manager
    exp_manager = ExperimentManager("4scale_longrun")
    
    # Create experiment plan
    config = create_experiment_config("extended", 100000)
    plan = exp_manager.create_experiment_plan(config)
    
    # Log initial milestone
    exp_manager.log_milestone("experiment_initialized", 
                             "4-scale FPN experiment setup completed")
    
    # Start system monitoring
    system_info = exp_manager.start_system_monitoring()
    exp_manager.append_to_monitor(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM READY")
    exp_manager.append_to_monitor(f"  GPU Memory: {system_info.get('gpu_memory_used_mb', 'N/A')}/{system_info.get('gpu_memory_total_mb', 'N/A')} MB")
    exp_manager.append_to_monitor(f"  GPU Utilization: {system_info.get('gpu_utilization_percent', 'N/A')}%")
    
    print(f"✅ Experiment initialized: {exp_manager.experiment_id}")
    print(f"📁 Directory: {exp_manager.experiment_dir}")
    print(f"📊 Monitor file: {exp_manager.experiment_dir}/training_monitor.txt")
#!/usr/bin/env python3
"""
Long-term Training Script for 4-scale FPN (12-24 hours)
Comprehensive experiment with extended dataset and detailed monitoring
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import signal

def signal_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    print(f"\n🛑 Received signal {signum}. Graceful shutdown initiated...")
    # Cleanup will be handled by the training script's own signal handlers
    sys.exit(0)

def setup_experiment_environment():
    """Set up environment for long-term training."""
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create experiment timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"4scale_longrun_{timestamp}"
    
    # Set up directories
    base_dir = Path("/home/oeoiewt/eTraM/rvt_eTram")
    experiment_dir = base_dir / "experiments" / experiment_id
    
    # Ensure experiment directory exists
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["checkpoints", "confusion_matrices", "logs", "results"]:
        (experiment_dir / subdir).mkdir(exist_ok=True)
    
    return experiment_id, experiment_dir

def create_training_command(experiment_dir, data_path="/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_extended_sample"):
    """Create the comprehensive training command."""
    
    # Base training command with extended dataset - simplified for compatibility
    cmd = [
        "python", "train.py",
        "model=rnndet",
        "dataset=gen4",  # Use gen4 config with custom path
        f"dataset.path={data_path}",
        "+experiment/gen4=default.yaml",
        "hardware.gpus=0",
        "batch_size.train=6",
        "batch_size.eval=2",
        "hardware.num_workers.train=4",
        "hardware.num_workers.eval=3",
        "training.max_epochs=30",  # Increased for longer training
        "+model.head.num_classes=8",  # Ensure 8 classes
    ]
    
    # Add basic W&B configuration (minimal to avoid override issues)
    wandb_configs = [
        "wandb.project_name=eTraM_4scale_FPN",
        f"+wandb.group_name=longterm_training_{datetime.now().strftime('%Y%m%d')}",
    ]
    
    cmd.extend(wandb_configs)
    
    return cmd

def setup_monitoring(experiment_dir):
    """Set up monitoring files and directories."""
    
    monitor_file = experiment_dir / "training_monitor.txt"
    system_log = experiment_dir / "logs" / "system_monitor.log"
    training_log = experiment_dir / "logs" / "training.log"
    
    # Initialize monitor file
    with open(monitor_file, 'w') as f:
        f.write(f"🚀 Long-term 4-scale FPN Training Started\n")
        f.write(f"📅 Start Time: {datetime.now().isoformat()}\n")
        f.write(f"📁 Experiment Directory: {experiment_dir}\n")
        f.write(f"🎯 Target: 12-24 hour comprehensive training\n")
        f.write(f"📊 Dataset: Extended sample (45 sequences, ~247k objects)\n")
        f.write(f"🏗️  Architecture: RVT + 4-scale FPN (strides: 4,8,16,32)\n")
        f.write(f"=" * 80 + "\n\n")
    
    # Initialize system log
    system_log.parent.mkdir(exist_ok=True)
    with open(system_log, 'w') as f:
        f.write(f"System monitoring started: {datetime.now().isoformat()}\n")
    
    return monitor_file, system_log, training_log

def log_system_stats(system_log_file):
    """Log current system statistics."""
    try:
        # GPU info
        gpu_result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True)
        
        if gpu_result.returncode == 0:
            gpu_data = gpu_result.stdout.strip().split('\n')[0].split(', ')
            gpu_used, gpu_total, gpu_util, gpu_temp = gpu_data
            
            # CPU and memory info
            cpu_result = subprocess.run(['cat', '/proc/loadavg'], capture_output=True, text=True)
            load_avg = cpu_result.stdout.strip().split()[0] if cpu_result.returncode == 0 else "N/A"
            
            memory_result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            if memory_result.returncode == 0:
                memory_lines = memory_result.stdout.strip().split('\n')
                memory_used = memory_lines[1].split()[2]
                memory_total = memory_lines[1].split()[1]
            else:
                memory_used = memory_total = "N/A"
            
            # Log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "gpu_memory_used_mb": int(gpu_used),
                "gpu_memory_total_mb": int(gpu_total),
                "gpu_utilization_percent": int(gpu_util),
                "gpu_temperature_c": int(gpu_temp),
                "cpu_load_avg": float(load_avg) if load_avg != "N/A" else None,
                "memory_used_mb": int(memory_used) if memory_used != "N/A" else None,
                "memory_total_mb": int(memory_total) if memory_total != "N/A" else None
            }
            
            with open(system_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            return log_entry
        
    except Exception as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        with open(system_log_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        return error_entry

def run_longterm_training():
    """Execute the long-term training with comprehensive monitoring."""
    
    print("🚀 Starting Long-term 4-scale FPN Training Setup...")
    
    # Set up experiment environment
    experiment_id, experiment_dir = setup_experiment_environment()
    
    # Set up monitoring
    monitor_file, system_log, training_log = setup_monitoring(experiment_dir)
    
    # Verify dataset exists
    data_path = "/home/oeoiewt/eTraM/rvt_eTram/data/gen4_cls8_extended_sample"
    if not Path(data_path).exists():
        print(f"❌ Extended dataset not found at: {data_path}")
        print("🔧 Please run 'python extended_sampling.py' first to create the dataset")
        return False
    
    # Create training command
    training_cmd = create_training_command(experiment_dir, data_path)
    
    print(f"✅ Experiment ID: {experiment_id}")
    print(f"📁 Experiment Directory: {experiment_dir}")
    print(f"📊 Monitor File: {monitor_file}")
    print(f"🖥️  System Log: {system_log}")
    
    # Log initial system state
    initial_stats = log_system_stats(system_log)
    
    # Append to monitor file
    with open(monitor_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING COMMAND PREPARED\n")
        f.write(f"Command: {' '.join(training_cmd)}\n")
        f.write(f"Initial GPU Memory: {initial_stats.get('gpu_memory_used_mb', 'N/A')}/{initial_stats.get('gpu_memory_total_mb', 'N/A')} MB\n")
        f.write(f"Initial GPU Utilization: {initial_stats.get('gpu_utilization_percent', 'N/A')}%\n")
        f.write(f"GPU Temperature: {initial_stats.get('gpu_temperature_c', 'N/A')}°C\n\n")
    
    print("\n🎯 Training Command:")
    print(" ".join(training_cmd))
    
    print(f"\n📋 Monitor Training Progress:")
    print(f"tail -f {monitor_file}")
    print(f"\n📊 System Monitoring:")
    print(f"tail -f {system_log}")
    
    print(f"\n🔥 Starting Long-term Training...")
    print(f"⏰ Expected Duration: 12-24 hours")
    print(f"📈 Dataset: 45 sequences, ~247k objects, 8 classes")
    print(f"🏗️  Architecture: 4-scale FPN with P1 features")
    
    # Change to working directory
    os.chdir("/home/oeoiewt/eTraM/rvt_eTram")
    
    # Execute training with real-time logging
    try:
        with open(monitor_file, 'a') as monitor_f, open(training_log, 'w') as training_f:
            
            monitor_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING STARTED\n")
            monitor_f.flush()
            
            # Start training process
            process = subprocess.Popen(
                training_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Real-time output monitoring
            start_time = time.time()
            last_system_log = time.time()
            
            for line in iter(process.stdout.readline, ''):
                current_time = time.time()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Write to training log
                training_f.write(f"[{timestamp}] {line}")
                training_f.flush()
                
                # Write important lines to monitor
                if any(keyword in line.lower() for keyword in [
                    'epoch', 'val_loss', 'train_loss', 'confusion', 'checkpoint', 
                    'error', 'warning', 'gpu', 'memory', 'step'
                ]):
                    monitor_f.write(f"[{timestamp}] {line}")
                    monitor_f.flush()
                
                # Log system stats every 5 minutes
                if current_time - last_system_log > 300:  # 5 minutes
                    log_system_stats(system_log)
                    last_system_log = current_time
                    
                    # Calculate elapsed time
                    elapsed_hours = (current_time - start_time) / 3600
                    monitor_f.write(f"[{timestamp}] MILESTONE: {elapsed_hours:.1f} hours elapsed\n")
                    monitor_f.flush()
                
                # Print real-time to console
                print(f"[{timestamp}] {line}", end='')
            
            # Wait for process completion
            return_code = process.wait()
            
            end_time = time.time()
            total_hours = (end_time - start_time) / 3600
            
            monitor_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING COMPLETED\n")
            monitor_f.write(f"Total Duration: {total_hours:.2f} hours\n")
            monitor_f.write(f"Return Code: {return_code}\n")
            
            if return_code == 0:
                print(f"\n✅ Training completed successfully!")
                print(f"⏱️  Total Duration: {total_hours:.2f} hours")
                print(f"📁 Results saved in: {experiment_dir}")
            else:
                print(f"\n❌ Training failed with return code: {return_code}")
                print(f"📁 Logs available in: {experiment_dir}")
            
            return return_code == 0
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        with open(monitor_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING INTERRUPTED BY USER\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed with exception: {e}")
        with open(monitor_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINING FAILED: {e}\n")
        return False

def create_screen_script():
    """Create a script for running training in screen session."""
    
    script_content = f'''#!/bin/bash
# Long-term Training Script for Screen Session

echo "🚀 Starting 4-scale FPN Long-term Training in Screen Session"
echo "📅 Started at: $(date)"

# Change to working directory
cd /home/oeoiewt/eTraM/rvt_eTram

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rvt

# Verify environment
echo "🐍 Python: $(which python)"
echo "📦 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the long-term training
python longterm_training.py

echo "✅ Training session completed at: $(date)"
'''
    
    script_path = Path("/home/oeoiewt/eTraM/rvt_eTram/run_longterm_training.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable
    
    print(f"📜 Screen script created: {script_path}")
    
    return script_path

if __name__ == "__main__":
    print("🔧 Long-term Training Setup")
    
    # Create screen script
    screen_script = create_screen_script()
    
    print(f"\n📋 To run in background with screen:")
    print(f"screen -S etram_4scale_longrun {screen_script}")
    print(f"\n📋 To attach to running session:")
    print(f"screen -r etram_4scale_longrun")
    print(f"\n📋 To detach from session: Ctrl+A then D")
    
    # Check if running in screen or direct mode
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        print(f"\n🔥 Starting training directly...")
        success = run_longterm_training()
        if success:
            print(f"\n🎉 Training completed successfully!")
        else:
            print(f"\n💥 Training encountered issues. Check logs for details.")
    else:
        print(f"\n✅ Setup complete! Use the screen command above to start training.")
        print(f"📁 All files ready for long-term training execution.")
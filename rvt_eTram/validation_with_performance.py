import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.performance_monitor import PerformanceMonitor


class PerformanceCallback(pl.Callback):
    """PyTorch Lightning callback for performance monitoring during validation."""
    
    def __init__(self, warmup_iterations: int = 10, enable_memory_profiling: bool = True):
        self.monitor = PerformanceMonitor(
            warmup_iterations=warmup_iterations,
            enable_memory_profiling=enable_memory_profiling
        )
        self.metrics = None
        
    def on_validation_start(self, trainer, pl_module):
        """Initialize performance monitoring at validation start."""
        self.monitor.start_session()
        print("🚀 Performance monitoring started...")
        
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch size and start timing."""
        self.current_batch_size = len(batch) if isinstance(batch, (list, tuple)) else batch.shape[0]
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record inference time after each batch."""
        # Note: This is called after the batch processing is complete
        # The actual timing is handled in the model's validation_step
        pass
        
    def on_validation_end(self, trainer, pl_module):
        """Calculate final metrics and save results."""
        try:
            self.metrics = self.monitor.end_session()
            self.monitor.print_summary(self.metrics)
            
            # Save performance metrics
            if hasattr(pl_module, 'experiment_id'):
                output_dir = f"validation_results/{pl_module.experiment_id}/"
            else:
                output_dir = "validation_results/latest/"
            
            os.makedirs(output_dir, exist_ok=True)
            self.monitor.save_metrics(self.metrics, f"{output_dir}/performance_metrics.json")
            print(f"💾 Performance metrics saved to {output_dir}/performance_metrics.json")
            
        except Exception as e:
            print(f"⚠️ Error calculating performance metrics: {e}")


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
    
    # Add experiment ID for performance logging
    checkpoint_path = Path(config.checkpoint)
    if 'experiments' in str(checkpoint_path):
        # Extract experiment ID from path like: experiments/patch2_sizeaware_100k/checkpoints/final_model.ckpt
        experiment_id = str(checkpoint_path).split('experiments/')[1].split('/')[0]
        module.experiment_id = experiment_id
    else:
        module.experiment_id = "unknown_experiment"

    # ---------------------
    # Performance Monitoring Setup
    # ---------------------
    enable_performance = config.get('enable_performance_monitoring', True)
    warmup_iterations = config.get('performance_warmup_iterations', 10)
    
    performance_callback = PerformanceCallback(
        warmup_iterations=warmup_iterations,
        enable_memory_profiling=True
    )

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]
    if enable_performance:
        callbacks.append(performance_callback)

    # ---------------------
    # Validation with Performance Monitoring
    # ---------------------
    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )

    print("🔄 Starting validation with performance monitoring...")
    
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
    
    # Additional performance summary
    if enable_performance and performance_callback.metrics:
        metrics = performance_callback.metrics
        
        print("\n" + "🎯 REAL-TIME SUITABILITY ASSESSMENT")
        print("=" * 50)
        
        assessments = [
            ("Traffic Monitoring", 10, "Good for detecting vehicles and pedestrians"),
            ("Autonomous Driving", 30, "Critical for real-time decision making"),
            ("Security Surveillance", 5, "Acceptable for monitoring applications"),
        ]
        
        for app_name, min_fps, description in assessments:
            status = "✅ SUITABLE" if metrics.fps >= min_fps else "❌ NOT SUITABLE"
            print(f"{app_name:20} (≥{min_fps:2d} FPS): {status}")
            print(f"                     {description}")
        
        print(f"\n💡 PERFORMANCE SUMMARY:")
        print(f"   Current FPS: {metrics.fps:.1f}")
        print(f"   Inference Time: {metrics.avg_inference_time*1000:.1f} ms")
        print(f"   Memory Usage: {metrics.gpu_memory_peak:.1f} GB GPU")
        
        # Comparison with previous experiments (if available)
        print(f"\n📊 EXPECTED IMPROVEMENTS (patch_size=2):")
        print(f"   Small objects: Better spatial resolution (2x improvement expected)")
        print(f"   Real-time performance: Should maintain 10+ FPS for practical use")
        print(f"   Memory efficiency: ~2x increase vs patch_size=4")


if __name__ == '__main__':
    main()
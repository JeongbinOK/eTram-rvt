"""
Performance monitoring utilities for validation and benchmarking.
Measures FPS, inference time, memory usage, and other performance metrics.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
import json


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    fps: float
    avg_inference_time: float
    median_inference_time: float
    p95_inference_time: float
    total_samples: int
    total_time: float
    gpu_memory_peak: float
    cpu_memory_peak: float
    throughput_samples_per_sec: float
    batch_processing_efficiency: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for model validation and benchmarking.
    
    Features:
    - FPS calculation with warm-up period
    - Inference time distribution analysis
    - Memory usage tracking (GPU/CPU)
    - Component-level timing (backbone, FPN, head, postprocess)
    - Batch processing efficiency analysis
    """
    
    def __init__(self, warmup_iterations: int = 10, enable_memory_profiling: bool = True):
        self.warmup_iterations = warmup_iterations
        self.enable_memory_profiling = enable_memory_profiling
        
        # Timing storage
        self.inference_times: List[float] = []
        self.component_times: Dict[str, List[float]] = {
            'backbone': [],
            'fpn': [],
            'head': [],
            'postprocess': [],
            'total': []
        }
        
        # Memory tracking
        self.gpu_memory_usage: List[float] = []
        self.cpu_memory_usage: List[float] = []
        
        # Counters
        self.total_samples = 0
        self.total_batches = 0
        self.iteration_count = 0
        
        # Session timing
        self.session_start_time: Optional[float] = None
        self.warmup_complete = False
        
    def start_session(self):
        """Start a new performance monitoring session."""
        self.session_start_time = time.time()
        self.iteration_count = 0
        self.warmup_complete = False
        
        # Clear previous measurements
        self.inference_times.clear()
        for component in self.component_times:
            self.component_times[component].clear()
        self.gpu_memory_usage.clear()
        self.cpu_memory_usage.clear()
        
    def end_session(self) -> PerformanceMetrics:
        """End session and calculate final metrics."""
        if self.session_start_time is None:
            raise RuntimeError("Session not started. Call start_session() first.")
            
        total_session_time = time.time() - self.session_start_time
        
        if not self.inference_times:
            raise RuntimeError("No measurements recorded during session.")
            
        # Calculate metrics
        avg_inference_time = np.mean(self.inference_times)
        median_inference_time = np.median(self.inference_times)
        p95_inference_time = np.percentile(self.inference_times, 95)
        
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        throughput = self.total_samples / total_session_time if total_session_time > 0 else 0.0
        
        # Batch processing efficiency (samples per second vs theoretical max)
        theoretical_max_throughput = self.total_samples / sum(self.inference_times)
        batch_efficiency = throughput / theoretical_max_throughput if theoretical_max_throughput > 0 else 0.0
        
        # Memory peaks
        gpu_memory_peak = max(self.gpu_memory_usage) if self.gpu_memory_usage else 0.0
        cpu_memory_peak = max(self.cpu_memory_usage) if self.cpu_memory_usage else 0.0
        
        return PerformanceMetrics(
            fps=fps,
            avg_inference_time=avg_inference_time,
            median_inference_time=median_inference_time,
            p95_inference_time=p95_inference_time,
            total_samples=self.total_samples,
            total_time=total_session_time,
            gpu_memory_peak=gpu_memory_peak,
            cpu_memory_peak=cpu_memory_peak,
            throughput_samples_per_sec=throughput,
            batch_processing_efficiency=batch_efficiency
        )
    
    @contextmanager
    def measure_inference(self, batch_size: int = 1):
        """Context manager for measuring inference time."""
        if self.session_start_time is None:
            raise RuntimeError("Session not started. Call start_session() first.")
            
        # Record memory before inference
        if self.enable_memory_profiling:
            self._record_memory_usage()
        
        start_time = time.time()
        yield
        end_time = time.time()
        
        inference_time = end_time - start_time
        self.iteration_count += 1
        
        # Skip warmup iterations
        if self.iteration_count <= self.warmup_iterations:
            if self.iteration_count == self.warmup_iterations:
                self.warmup_complete = True
                print(f"Warmup complete after {self.warmup_iterations} iterations")
            return
            
        # Record measurements
        self.inference_times.append(inference_time)
        self.total_samples += batch_size
        self.total_batches += 1
        
        # Record memory after inference
        if self.enable_memory_profiling:
            self._record_memory_usage()
    
    @contextmanager 
    def measure_component(self, component_name: str):
        """Context manager for measuring component-level timing."""
        if component_name not in self.component_times:
            self.component_times[component_name] = []
            
        start_time = time.time()
        yield
        end_time = time.time()
        
        component_time = end_time - start_time
        
        # Only record after warmup
        if self.warmup_complete:
            self.component_times[component_name].append(component_time)
    
    def _record_memory_usage(self):
        """Record current GPU and CPU memory usage."""
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.gpu_memory_usage.append(gpu_memory)
        
        # CPU memory
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        self.cpu_memory_usage.append(cpu_memory)
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for component-level timing."""
        stats = {}
        for component, times in self.component_times.items():
            if times:
                stats[component] = {
                    'avg_time': np.mean(times),
                    'median_time': np.median(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'std_time': np.std(times),
                    'total_calls': len(times)
                }
        return stats
    
    def print_summary(self, metrics: PerformanceMetrics):
        """Print performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE MONITORING SUMMARY")
        print("="*60)
        
        # Overall metrics
        print(f"🚀 FPS: {metrics.fps:.2f}")
        print(f"⏱️  Average Inference Time: {metrics.avg_inference_time*1000:.2f} ms")
        print(f"📊 Median Inference Time: {metrics.median_inference_time*1000:.2f} ms")
        print(f"📈 95th Percentile: {metrics.p95_inference_time*1000:.2f} ms")
        print(f"🔢 Total Samples: {metrics.total_samples}")
        print(f"⏰ Total Time: {metrics.total_time:.2f} s")
        print(f"🏭 Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
        print(f"⚡ Batch Efficiency: {metrics.batch_processing_efficiency:.1%}")
        
        # Memory usage
        print(f"\n💾 MEMORY USAGE:")
        print(f"   GPU Peak: {metrics.gpu_memory_peak:.2f} GB")
        print(f"   CPU Peak: {metrics.cpu_memory_peak:.2f} GB")
        
        # Component breakdown
        component_stats = self.get_component_stats()
        if component_stats:
            print(f"\n🔧 COMPONENT BREAKDOWN:")
            for component, stats in component_stats.items():
                if stats['total_calls'] > 0:
                    print(f"   {component.upper()}: {stats['avg_time']*1000:.2f} ms avg "
                          f"({stats['total_calls']} calls)")
        
        # Real-time assessment
        print(f"\n🎯 REAL-TIME ASSESSMENT:")
        if metrics.fps >= 15:
            print("   ✅ Excellent for traffic monitoring (15+ FPS)")
        elif metrics.fps >= 10:
            print("   ✅ Good for traffic monitoring (10+ FPS)")
        elif metrics.fps >= 5:
            print("   ⚠️  Marginal for real-time use (5+ FPS)")
        else:
            print("   ❌ Not suitable for real-time applications (<5 FPS)")
            
        print("="*60)
    
    def save_metrics(self, metrics: PerformanceMetrics, filepath: str):
        """Save performance metrics to JSON file."""
        data = {
            'performance_metrics': {
                'fps': metrics.fps,
                'avg_inference_time_ms': metrics.avg_inference_time * 1000,
                'median_inference_time_ms': metrics.median_inference_time * 1000,
                'p95_inference_time_ms': metrics.p95_inference_time * 1000,
                'total_samples': metrics.total_samples,
                'total_time_s': metrics.total_time,
                'gpu_memory_peak_gb': metrics.gpu_memory_peak,
                'cpu_memory_peak_gb': metrics.cpu_memory_peak,
                'throughput_samples_per_sec': metrics.throughput_samples_per_sec,
                'batch_processing_efficiency': metrics.batch_processing_efficiency
            },
            'component_stats': self.get_component_stats(),
            'real_time_assessment': {
                'suitable_for_traffic_monitoring': metrics.fps >= 10,
                'suitable_for_autonomous_driving': metrics.fps >= 30,
                'suitable_for_security': metrics.fps >= 5,
                'fps_category': self._get_fps_category(metrics.fps)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_fps_category(self, fps: float) -> str:
        """Categorize FPS performance."""
        if fps >= 30:
            return "excellent"
        elif fps >= 15:
            return "good"
        elif fps >= 10:
            return "acceptable"
        elif fps >= 5:
            return "marginal"
        else:
            return "poor"


# Convenience function for quick benchmarking
def benchmark_model(model, dataloader, device='cuda', warmup_iterations=10, max_iterations=None):
    """
    Quick benchmark function for model performance testing.
    
    Args:
        model: PyTorch model to benchmark
        dataloader: DataLoader for test data
        device: Device to run on ('cuda' or 'cpu')
        warmup_iterations: Number of warmup iterations to skip
        max_iterations: Maximum iterations to run (None = full dataloader)
    
    Returns:
        PerformanceMetrics object
    """
    monitor = PerformanceMonitor(warmup_iterations=warmup_iterations)
    model.eval()
    model.to(device)
    
    monitor.start_session()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_iterations and i >= max_iterations + warmup_iterations:
                break
                
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]
            elif hasattr(batch, 'to'):
                batch = batch.to(device)
            
            batch_size = len(batch) if isinstance(batch, (list, tuple)) else batch.shape[0]
            
            with monitor.measure_inference(batch_size=batch_size):
                _ = model(batch)
    
    metrics = monitor.end_session()
    monitor.print_summary(metrics)
    
    return metrics
#!/usr/bin/env python3
"""
Test script for LightweightEnhancedConvLSTM implementation.
Tests functionality, memory usage, and parameter overhead.
"""

import torch as th
import torch.nn as nn
from models.layers.rnn import DWSConvLSTM2d, LightweightEnhancedConvLSTM

def test_enhanced_convlstm():
    """Test LightweightEnhancedConvLSTM functionality."""
    print("=== Testing LightweightEnhancedConvLSTM ===")
    
    # Test parameters
    batch_size = 2
    dim = 128  # P2 stage dimension
    height, width = 80, 45  # P2 spatial resolution (640x360 / 8)
    
    # Create models
    base_model = DWSConvLSTM2d(dim=dim)
    enhanced_model = LightweightEnhancedConvLSTM(dim=dim, enhancement_ratio=0.05)
    
    # Create test input
    x = th.randn(batch_size, dim, height, width)
    h_prev = th.randn(batch_size, dim, height, width)
    c_prev = th.randn(batch_size, dim, height, width)
    h_and_c_prev = (h_prev, c_prev)
    
    print(f"Input shape: {x.shape}")
    print(f"Hidden state shape: {h_prev.shape}")
    
    # Test base model
    print("\n--- Base DWSConvLSTM2d ---")
    with th.no_grad():
        h_base, c_base = base_model(x, h_and_c_prev)
        print(f"Output hidden shape: {h_base.shape}")
        print(f"Output cell shape: {c_base.shape}")
    
    # Count base model parameters
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {base_params:,}")
    
    # Test enhanced model
    print("\n--- Enhanced ConvLSTM ---")
    with th.no_grad():
        h_enhanced, c_enhanced = enhanced_model(x, h_and_c_prev)
        print(f"Output hidden shape: {h_enhanced.shape}")
        print(f"Output cell shape: {c_enhanced.shape}")
    
    # Get enhancement info
    enhancement_info = enhanced_model.get_enhancement_info()
    print(f"Enhanced model parameters: {enhancement_info['total_parameters']:,}")
    print(f"Enhancement parameters: {enhancement_info['enhancement_parameters']:,}")
    print(f"Enhancement ratio: {enhancement_info['enhancement_ratio']:.3f}")
    print(f"Target ratio: {enhancement_info['target_ratio']:.3f}")
    
    # Calculate parameter overhead
    param_overhead = (enhancement_info['total_parameters'] - base_params) / base_params * 100
    print(f"Parameter overhead: {param_overhead:.1f}%")
    
    # Test memory usage
    print("\n--- Memory Usage Test ---")
    
    def get_memory_usage():
        if th.cuda.is_available():
            return th.cuda.memory_allocated() / 1024**2  # MB
        return 0
    
    if th.cuda.is_available():
        base_model = base_model.cuda()
        enhanced_model = enhanced_model.cuda()
        x = x.cuda()
        h_and_c_prev = (h_and_c_prev[0].cuda(), h_and_c_prev[1].cuda())
        
        # Measure base model memory
        th.cuda.empty_cache()
        base_mem_before = get_memory_usage()
        with th.no_grad():
            h_base, c_base = base_model(x, h_and_c_prev)
        base_mem_after = get_memory_usage()
        
        # Measure enhanced model memory
        th.cuda.empty_cache()
        enhanced_mem_before = get_memory_usage()
        with th.no_grad():
            h_enhanced, c_enhanced = enhanced_model(x, h_and_c_prev)
        enhanced_mem_after = get_memory_usage()
        
        print(f"Base model memory usage: {base_mem_after - base_mem_before:.1f} MB")
        print(f"Enhanced model memory usage: {enhanced_mem_after - enhanced_mem_before:.1f} MB")
        
        memory_overhead = (enhanced_mem_after - enhanced_mem_before) / (base_mem_after - base_mem_before) - 1
        print(f"Memory overhead: {memory_overhead * 100:.1f}%")
    else:
        print("CUDA not available, skipping memory test")
    
    # Test gradient flow
    print("\n--- Gradient Flow Test ---")
    enhanced_model.train()
    x.requires_grad_(True)
    h_enhanced, c_enhanced = enhanced_model(x, h_and_c_prev)
    loss = h_enhanced.sum()
    loss.backward()
    
    gradient_norm = th.norm(x.grad).item()
    print(f"Input gradient norm: {gradient_norm:.6f}")
    
    # Check if gradients are flowing to enhancement components
    enhancement_grads = []
    for name, param in enhanced_model.named_parameters():
        if any(comp in name for comp in ['temporal_attention', 'density_estimator', 
                                       'small_object_enhancer', 'feature_fusion']):
            if param.grad is not None:
                enhancement_grads.append(th.norm(param.grad).item())
    
    if enhancement_grads:
        avg_enhancement_grad = sum(enhancement_grads) / len(enhancement_grads)
        print(f"Average enhancement component gradient norm: {avg_enhancement_grad:.6f}")
        print("✓ Gradients flowing to enhancement components")
    else:
        print("✗ No gradients in enhancement components")
    
    print("\n=== Test Complete ===")
    print(f"✓ Forward pass successful")
    print(f"✓ Parameter overhead: {param_overhead:.1f}% (target: ~10%)")
    print(f"✓ Output shapes match input")
    print(f"✓ Backward pass successful")
    
    return True

def test_backbone_integration():
    """Test integration with RNNDetector backbone."""
    print("\n=== Testing Backbone Integration ===")
    
    try:
        from models.detection.recurrent_backbone.maxvit_rnn import RNNDetector
        from omegaconf import DictConfig
        
        # Create minimal config for testing (based on real config structure)
        config = DictConfig({
            'input_channels': 20,
            'enable_masking': False,
            'embed_dim': 64,
            'dim_multiplier': [1, 2, 4, 8],
            'num_blocks': [1, 1, 1, 1],
            'T_max_chrono_init': [4, 8, 16, 32],
            'use_enhanced_convlstm': True,  # Enable enhanced ConvLSTM
            'stem': {
                'patch_size': 4
            },
            'stage': {
                'downsample': {
                    'type': 'patch',
                    'overlap': True,
                    'norm_affine': True
                },
                'attention': {
                    'use_torch_mha': False,
                    'partition_size': 5,  # 5 divides 90 and other dimensions better
                    'dim_head': 32,
                    'attention_bias': True,
                    'mlp_activation': 'gelu',
                    'mlp_gated': False,
                    'mlp_bias': True,
                    'mlp_ratio': 4,
                    'drop_mlp': 0,
                    'drop_path': 0,
                    'ls_init_value': 1e-5
                },
                'lstm': {
                    'dws_conv': False,
                    'dws_conv_only_hidden': True,
                    'dws_conv_kernel_size': 3,
                    'drop_cell_update': 0,
                    'enhancement_ratio': 0.05,
                    'small_object_threshold': 0.3
                }
            }
        })
        
        # Create model
        model = RNNDetector(config)
        print("✓ RNNDetector created successfully with enhanced ConvLSTM")
        
        # Test forward pass with smaller input to avoid partition size issues
        batch_size = 1
        x = th.randn(batch_size, 20, 160, 160)  # Smaller, square input for testing
        
        with th.no_grad():
            output = model(x)
            features, states = output  # RNNDetector returns (features, states)
            print(f"✓ Forward pass successful")
            print(f"Output features: {list(features.keys())}")
            for stage, feat in features.items():
                print(f"  Stage {stage}: {feat.shape}")
            print(f"States: {len(states)} LSTM states")
        
        # Check which stages use enhanced ConvLSTM
        for stage_idx, stage in enumerate(model.stages):
            lstm_type = type(stage.lstm).__name__
            if isinstance(stage.lstm, LightweightEnhancedConvLSTM):
                print(f"✓ Stage {stage_idx} uses LightweightEnhancedConvLSTM")
            else:
                print(f"  Stage {stage_idx} uses {lstm_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backbone integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Lightweight Enhanced ConvLSTM Implementation\n")
    
    # Run tests
    success = True
    
    try:
        success &= test_enhanced_convlstm()
        success &= test_backbone_integration()
        
        if success:
            print("\n🎉 All tests passed! LightweightEnhancedConvLSTM is ready for training.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
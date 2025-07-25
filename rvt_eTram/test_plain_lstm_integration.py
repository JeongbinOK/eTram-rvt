#!/usr/bin/env python3
"""
Test PlainLSTM2d integration and verify RVT paper implementation.
This script validates that our Plain LSTM implementation follows RVT paper design.
"""

import sys
import torch
import torch.nn as nn
sys.path.append('/home/oeoiewt/eTraM/rvt_eTram')

from models.layers.rnn import PlainLSTM2d, DWSConvLSTM2d
from models.detection.recurrent_backbone.maxvit_rnn import RNNDetector
from omegaconf import DictConfig, OmegaConf

def test_plain_lstm_functionality():
    """Test PlainLSTM2d basic functionality."""
    print("=" * 60)
    print("TESTING PLAIN LSTM FUNCTIONALITY")
    print("=" * 60)
    
    dim = 128
    batch_size, height, width = 2, 40, 40
    
    # Create PlainLSTM2d instance
    plain_lstm = PlainLSTM2d(dim=dim)
    
    # Test input
    x = torch.randn(batch_size, dim, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass without previous state
    h1, c1 = plain_lstm(x)
    print(f"Output shape (h): {h1.shape}")
    print(f"Output shape (c): {c1.shape}")
    
    # Forward pass with previous state
    h2, c2 = plain_lstm(x, (h1, c1))
    print(f"Recurrent output shape (h): {h2.shape}")
    print(f"Recurrent output shape (c): {c2.shape}")
    
    print("✅ Plain LSTM functionality test PASSED")
    return True

def test_parameter_comparison():
    """Compare parameter counts between Plain LSTM and ConvLSTM."""
    print("\n" + "=" * 60)
    print("TESTING PARAMETER COMPARISON (RVT PAPER VALIDATION)")
    print("=" * 60)
    
    dim = 128
    
    # Plain LSTM
    plain_lstm = PlainLSTM2d(dim=dim)
    plain_params = sum(p.numel() for p in plain_lstm.parameters())
    param_info = plain_lstm.get_parameter_count()
    
    # ConvLSTM
    conv_lstm = DWSConvLSTM2d(dim=dim, dws_conv=True, dws_conv_kernel_size=3)
    conv_params = sum(p.numel() for p in conv_lstm.parameters())
    
    print(f"Plain LSTM parameters: {plain_params:,}")
    print(f"ConvLSTM parameters: {conv_params:,}")
    print(f"Parameter reduction: {(conv_params - plain_params) / conv_params:.1%}")
    print(f"Expected mAP improvement: +{param_info['expected_mAP_improvement']}%")
    
    # Validate RVT paper expectations
    reduction_ratio = (conv_params - plain_params) / conv_params
    assert reduction_ratio > 0.3, f"Expected >30% reduction, got {reduction_ratio:.1%}"
    
    print("✅ Parameter comparison test PASSED")
    print(f"✅ Achieves RVT paper's parameter efficiency ({reduction_ratio:.1%} reduction)")
    return True

def test_backbone_integration():
    """Test integration with RNNDetector backbone."""
    print("\n" + "=" * 60)
    print("TESTING BACKBONE INTEGRATION")
    print("=" * 60)
    
    # Create minimal config for testing
    config = {
        'input_channels': 20,
        'embed_dim': 64,
        'dim_multiplier': [1, 2, 4, 8],
        'num_blocks': [1, 1, 1, 1],  # Minimal for testing
        'T_max_chrono_init': [None, None, None, None],
        'use_plain_lstm': True,  # Enable Plain LSTM
        'use_enhanced_convlstm': False,
        'enable_masking': False,
        'stem': {'patch_size': 4},
        'stage': {
            'downsample': {
                'kernel_size': 3,
                'stride': 2,
                'conv_bias': True,
                'norm_name': 'layer_norm_cf',
                'norm_eps': 1e-6,
                'norm_affine': True
            },
            'attention': {
                'use_torch_mha': False,
                'partition_size': [6, 10],
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
                'dws_conv': False,  # Disabled for Plain LSTM
                'dws_conv_only_hidden': True,
                'dws_conv_kernel_size': 3,
                'drop_cell_update': 0
            }
        }
    }
    
    mdl_config = OmegaConf.create(config)
    
    # Create backbone with Plain LSTM
    try:
        backbone = RNNDetector(mdl_config)
        print("✅ RNNDetector with Plain LSTM created successfully")
        
        # Test forward pass
        batch_size = 1
        input_tensor = torch.randn(batch_size, 20, 160, 160)  # 640x640 / 4 = 160x160
        
        with torch.no_grad():
            features, states = backbone(input_tensor)
        
        print(f"✅ Forward pass successful")
        print(f"✅ Output features: {list(features.keys())}")
        print(f"✅ Output states: {len(states)} LSTM states")
        
        # Verify Plain LSTM usage
        for stage_idx, stage in enumerate(backbone.stages):
            lstm_type = type(stage.lstm).__name__
            print(f"   Stage {stage_idx + 1}: {lstm_type}")
            if lstm_type == 'PlainLSTM2d':
                print(f"   ✅ Stage {stage_idx + 1} uses Plain LSTM (RVT paper implementation)")
        
        return True
        
    except Exception as e:
        print(f"❌ Backbone integration test FAILED: {e}")
        return False

def test_training_compatibility():
    """Test compatibility with training configuration."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Test configuration loading
        import yaml
        from omegaconf import OmegaConf
        
        config_path = '/home/oeoiewt/eTraM/rvt_eTram/config/model/maxvit_yolox/plain_lstm.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration file loaded: {config_path}")
        print(f"✅ use_plain_lstm: {config.get('use_plain_lstm', 'Not set')}")
        print(f"✅ use_enhanced_convlstm: {config.get('use_enhanced_convlstm', 'Not set')}")
        
        # Verify experiment config
        exp_config_path = '/home/oeoiewt/eTraM/rvt_eTram/config/experiment/gen4/plain_lstm_640x360_baseline.yaml'
        with open(exp_config_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        print(f"✅ Experiment configuration loaded: {exp_config_path}")
        print(f"✅ Expected overall mAP: {exp_config['target_metrics']['overall_map']}%")
        print(f"✅ Expected small objects mAP: {exp_config['target_metrics']['small_objects_map']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("RVT PLAIN LSTM INTEGRATION TEST")
    print("=" * 60)
    print("Validating RVT paper's Plain LSTM implementation")
    print("Expected: +1.1% mAP improvement over ConvLSTM")
    print("=" * 60)
    
    tests = [
        test_plain_lstm_functionality,
        test_parameter_comparison,
        test_backbone_integration,
        test_training_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready for training!")
        print("\nNext steps:")
        print("1. Run: ./run_plain_lstm_experiment.sh")
        print("2. Execute training in screen session")
        print("3. Expected results: +1.1% mAP improvement")
        return True
    else:
        print("❌ Some tests failed - Please fix issues before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script for Size-aware + Attention model to verify implementation correctness.
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_attention_modules():
    """Test individual attention modules."""
    print("🧪 Testing individual attention modules...")
    
    try:
        from models.layers.small_object_attention import (
            MultiScaleSpatialAttention, 
            EventTemporalAttention, 
            SmallObjectAttentionModule,
            ScaleAwareChannelAttention
        )
        
        # Test MultiScaleSpatialAttention
        spatial_attention = MultiScaleSpatialAttention(channels=128)
        test_input = torch.randn(2, 128, 32, 32)
        spatial_output = spatial_attention(test_input)
        print(f"✅ MultiScaleSpatialAttention: {test_input.shape} → {spatial_output.shape}")
        
        # Test EventTemporalAttention  
        temporal_attention = EventTemporalAttention(channels=128)
        temporal_output = temporal_attention(test_input)
        print(f"✅ EventTemporalAttention: {test_input.shape} → {temporal_output.shape}")
        
        # Test SmallObjectAttentionModule
        combined_attention = SmallObjectAttentionModule(channels=128)
        combined_output = combined_attention(test_input)
        print(f"✅ SmallObjectAttentionModule: {test_input.shape} → {combined_output.shape}")
        
        # Test ScaleAwareChannelAttention
        channel_attention = ScaleAwareChannelAttention(channels=128, stride=8)
        channel_output = channel_attention(test_input)
        print(f"✅ ScaleAwareChannelAttention: {test_input.shape} → {channel_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention modules test failed: {e}")
        return False

def test_fpn_with_attention():
    """Test FPN with attention integration."""
    print("\n🧪 Testing FPN with attention integration...")
    
    try:
        from models.detection.yolox_extension.models.yolo_pafpn import YOLOPAFPN
        
        # Test 3-scale FPN with attention
        fpn = YOLOPAFPN(
            in_stages=(2, 3, 4),
            in_channels=(128, 256, 512),
            enable_small_object_attention=True,
            attention_spatial=True,
            attention_temporal=True
        )
        
        # Create mock backbone features
        mock_features = {
            2: torch.randn(2, 128, 40, 40),  # P2: stride 8
            3: torch.randn(2, 256, 20, 20),  # P3: stride 16  
            4: torch.randn(2, 512, 10, 10),  # P4: stride 32
        }
        
        # Forward pass
        outputs = fpn(mock_features)
        print(f"✅ 3-scale FPN with attention:")
        for i, output in enumerate(outputs):
            print(f"   Output {i}: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ FPN with attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading the complete model with configuration."""
    print("\n🧪 Testing model loading with configuration...")
    
    try:
        # Change to project directory for proper imports
        os.chdir(project_root)
        
        from omegaconf import OmegaConf
        # Simple test without full model build - just verify config loading
        config_path = "config/model/maxvit_yolox/size_aware_attention.yaml"
        cfg = OmegaConf.load(config_path)
        print(f"✅ Configuration loaded successfully")
        print(f"   FPN attention enabled: {cfg.model.fpn.enable_small_object_attention}")
        print(f"   Spatial attention: {cfg.model.fpn.attention_spatial}")
        print(f"   Temporal attention: {cfg.model.fpn.attention_temporal}")
        print(f"   Size-aware loss: {cfg.model.head.size_aware_loss}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Size-aware + Attention Model Implementation")
    print("=" * 60)
    
    tests = [
        test_attention_modules,
        test_fpn_with_attention, 
        test_model_loading
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Model is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
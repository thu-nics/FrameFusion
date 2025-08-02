#!/usr/bin/env python3
"""
Regression test for rope_deltas AttributeError issue (#7)

This test ensures that Qwen2VL models have the rope_deltas attribute properly
initialized when FrameFusion is applied, preventing AttributeError during forward pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
from framefusion.interface import apply_framefusion

def test_qwen2vl_rope_deltas_initialization():
    """
    Test that rope_deltas attribute is properly initialized when applying FrameFusion.
    
    This test verifies the fix for issue #7 where Qwen2VL models would throw
    AttributeError: 'Qwen2VLForConditionalGeneration' object has no attribute 'rope_deltas'
    """
    print("Testing Qwen2VL rope_deltas initialization...")
    
    # Create a minimal Qwen2VL model configuration
    config = Qwen2VLConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=512,
        vision_config={
            "depth": 1,
            "embed_dim": 256,
            "mlp_ratio": 2,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2
        },
        video_token_id=152084,
        image_token_id=151646,
    )
    
    try:
        # Create model
        model = Qwen2VLForConditionalGeneration(config)
        
        # Verify rope_deltas doesn't exist initially
        assert not hasattr(model, 'rope_deltas'), "Model should not have rope_deltas initially"
        
        # Apply FrameFusion - this should initialize rope_deltas
        apply_framefusion(
            model,
            cost=0.3,
            similarity_lower_bound=0.5,
            ratio_lower_bound=0.1,
        )
        
        # Verify rope_deltas is now initialized
        assert hasattr(model, 'rope_deltas'), "Model should have rope_deltas after applying FrameFusion"
        assert model.rope_deltas is None, "rope_deltas should be initialized to None"
        
        print("✓ Test passed - rope_deltas properly initialized")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen2vl_rope_deltas_initialization()
    exit(0 if success else 1)
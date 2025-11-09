#!/usr/bin/env python3
"""
GPU Test Script for RTX 2060

Checks:
1. CUDA availability
2. GPU memory
3. Simple training step
"""

import torch
import sys

def test_gpu():
    print("\n" + "="*60)
    print("🎮 RTX 2060 GPU Test")
    print("="*60 + "\n")
    
    # Test 1: CUDA
    print("Test 1: CUDA Availability")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA not available!")
        print("Possible reasons:")
        print("  1. NVIDIA drivers not installed")
        print("  2. PyTorch CPU version installed")
        print("  3. CUDA toolkit not installed")
        print("\nInstall GPU PyTorch:")
        print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    print()
    
    # Test 2: GPU Info
    print("Test 2: GPU Information")
    gpu_count = torch.cuda.device_count()
    print(f"  GPU count: {gpu_count}")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name}")
        print(f"  Total memory: {memory_total:.1f} GB")
    print()
    
    # Test 3: Memory Test
    print("Test 3: GPU Memory Test")
    device = torch.device("cuda:0")
    
    try:
        # Allocate tensors to test memory
        x = torch.randn(2, 3, 224, 224).to(device)  # Batch of 2 images
        print(f"  ✓ Allocated image tensor: {x.shape}")
        
        y = torch.randn(2, 512, 768).to(device)  # Text features
        print(f"  ✓ Allocated text tensor: {y.shape}")
        
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"  Memory allocated: {memory_allocated:.1f} MB")
        print(f"  Memory reserved: {memory_reserved:.1f} MB")
        
        # Clear memory
        del x, y
        torch.cuda.empty_cache()
        print(f"  ✓ Memory cleared successfully")
        
    except Exception as e:
        print(f"  ✗ Memory test failed: {e}")
        return False
    
    print()
    
    # Test 4: Simple operation
    print("Test 4: GPU Computation Test")
    try:
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)
        c = torch.matmul(a, b)
        print(f"  ✓ Matrix multiplication: {c.shape}")
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            d = torch.matmul(a, b)
        print(f"  ✓ Mixed precision (FP16) works")
        
        del a, b, c, d
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Computation test failed: {e}")
        return False
    
    print()
    
    # Summary
    print("="*60)
    print("✅ ALL GPU TESTS PASSED!")
    print("="*60)
    print()
    print("Your RTX 2060 is ready for training!")
    print()
    print("Recommended settings for 6GB GPU:")
    print("  - Batch size: 2-4")
    print("  - Image size: 224x224")
    print("  - Mixed precision: enabled (FP16)")
    print("  - Gradient checkpointing: enabled")
    print()
    print("Next step:")
    print("  python src/train.py --config configs/rtx2060_config.yaml")
    print()
    
    return True


if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)


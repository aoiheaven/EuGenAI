#!/usr/bin/env python3
"""
Sanity Check Script

Tests that all core functionality works without data.
Helps identify crashes before actual training.

Usage:
    python scripts/sanity_check.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from PIL import Image

print("=" * 70)
print("SANITY CHECK: Medical Multimodal Chain-of-Thought Framework")
print("=" * 70)
print()

# Test 1: Import all modules
print("[1/8] Testing imports...")
try:
    from dataset import MedicalChainOfThoughtDataset, collate_fn
    from model import MedicalMultimodalCoT
    from utils import DiagnosisLabelEncoder, TextProcessor, validate_config
    from inference import MedicalCoTInference
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model initialization
print("\n[2/8] Testing model initialization...")
try:
    model_config = {
        'image_encoder_name': 'vit_base_patch16_224',
        'text_encoder_name': 'bert-base-uncased',
        'img_size': 224,  # Small size for testing
        'hidden_dim': 768,
        'num_cot_steps': 5,
        'num_heads': 4,
        'num_decoder_layers': 2,
        'dropout': 0.1,
        'num_diagnosis_classes': 10,
    }
    model = MedicalMultimodalCoT(**model_config)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 3: Forward pass with dummy data
print("\n[3/8] Testing model forward pass...")
try:
    batch_size = 2
    img_size = 224
    seq_len = 128
    num_steps = 5
    
    dummy_inputs = {
        'images': torch.randn(batch_size, 3, img_size, img_size),
        'text_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'text_attention_mask': torch.ones(batch_size, seq_len),
        'cot_step_input_ids': torch.randint(0, 1000, (batch_size, num_steps, seq_len)),
        'cot_step_attention_mask': torch.ones(batch_size, num_steps, seq_len),
        'cot_step_regions': torch.rand(batch_size, num_steps, 4) * 100,
        'cot_num_steps': torch.tensor([num_steps, num_steps]),
    }
    
    model.eval()
    with torch.no_grad():
        outputs = model(**dummy_inputs)
    
    assert 'diagnosis_logits' in outputs
    assert 'confidence' in outputs
    assert outputs['diagnosis_logits'].shape == (batch_size, 10)
    assert outputs['confidence'].shape == (batch_size, 1)
    print("✓ Forward pass successful")
    print(f"  - Diagnosis logits shape: {outputs['diagnosis_logits'].shape}")
    print(f"  - Confidence shape: {outputs['confidence'].shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: DiagnosisLabelEncoder
print("\n[4/8] Testing DiagnosisLabelEncoder...")
try:
    encoder = DiagnosisLabelEncoder()
    
    # Add some labels
    labels = ["Pneumonia", "Heart Failure", "Normal", "COVID-19"]
    for label in labels:
        encoder.add_label(label)
    
    # Test encoding/decoding
    label_id = encoder.encode("Pneumonia")
    decoded = encoder.decode(label_id)
    
    assert decoded == "Pneumonia"
    assert encoder.num_classes == 4
    
    # Test batch encoding
    batch_labels = ["Heart Failure", "Normal", "Pneumonia"]
    batch_ids = encoder.encode_batch(batch_labels)
    
    assert batch_ids.shape == (3,)
    print("✓ Label encoder working")
    print(f"  - Vocabulary size: {encoder.num_classes}")
except Exception as e:
    print(f"✗ Label encoder failed: {e}")
    sys.exit(1)

# Test 5: TextProcessor
print("\n[5/8] Testing TextProcessor...")
try:
    processor = TextProcessor(max_length=128)
    
    # Test clinical text encoding
    encoded = processor.encode_clinical_text(
        history="Patient has fever",
        physical_exam="Temperature 38.5C",
        lab_results="WBC elevated"
    )
    
    assert 'input_ids' in encoded
    assert 'attention_mask' in encoded
    assert encoded['input_ids'].shape == (128,)
    
    # Test reasoning steps encoding
    steps = [
        {'action': 'Examine image', 'observation': 'See opacity'},
        {'action': 'Check labs', 'observation': 'WBC high', 'reasoning': 'Infection likely'}
    ]
    encoded_steps = processor.encode_reasoning_steps(steps, max_steps=5)
    
    assert encoded_steps['input_ids'].shape == (5, 128)
    assert encoded_steps['num_steps'] == 2
    
    print("✓ Text processor working")
except Exception as e:
    print(f"✗ Text processor failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Config validation
print("\n[6/8] Testing config validation...")
try:
    valid_config = {
        'model': {
            'image_encoder_name': 'vit_base_patch16_224',
            'text_encoder_name': 'bert-base-uncased',
            'hidden_dim': 768,
        },
        'dataset': {
            'data_root': 'data',
            'train_file': 'data/train.json',
            'val_file': 'data/val.json',
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 100,
            'learning_rate': 1e-4,
        }
    }
    
    validate_config(valid_config)
    print("✓ Config validation working")
    
    # Test invalid config
    invalid_config = {'model': {}}  # Missing required fields
    try:
        validate_config(invalid_config)
        print("✗ Should have raised ValueError")
    except ValueError:
        print("✓ Invalid config properly rejected")
        
except Exception as e:
    print(f"✗ Config validation failed: {e}")
    sys.exit(1)

# Test 7: Loss computation
print("\n[7/8] Testing loss computation...")
try:
    import torch.nn as nn
    
    criterion_diagnosis = nn.CrossEntropyLoss()
    criterion_confidence = nn.MSELoss()
    
    # Dummy predictions and targets
    pred_logits = torch.randn(4, 10)
    target_labels = torch.randint(0, 10, (4,))
    pred_confidence = torch.rand(4, 1)
    target_confidence = torch.rand(4)
    
    # Compute losses
    diag_loss = criterion_diagnosis(pred_logits, target_labels)
    conf_loss = criterion_confidence(pred_confidence.squeeze(), target_confidence)
    
    total_loss = diag_loss + 0.5 * conf_loss
    
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)
    
    print("✓ Loss computation working")
    print(f"  - Diagnosis loss: {diag_loss.item():.4f}")
    print(f"  - Confidence loss: {conf_loss.item():.4f}")
    print(f"  - Total loss: {total_loss.item():.4f}")
except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    sys.exit(1)

# Test 8: Gradient flow
print("\n[8/8] Testing gradient flow...")
try:
    model = MedicalMultimodalCoT(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    outputs = model(**dummy_inputs)
    
    # Compute loss
    dummy_labels = torch.randint(0, 10, (batch_size,))
    dummy_confidence = torch.rand(batch_size)
    
    loss = criterion_diagnosis(outputs['diagnosis_logits'], dummy_labels)
    loss += criterion_confidence(outputs['confidence'].squeeze(), dummy_confidence)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed!"
    
    # Optimizer step
    optimizer.step()
    
    print("✓ Gradient flow working")
    print("  - Gradients computed successfully")
    print("  - Optimizer step completed")
except Exception as e:
    print(f"✗ Gradient flow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL SANITY CHECKS PASSED! ✓")
print("=" * 70)
print("\nThe code is ready for training (once you have data).")
print("\nNext steps:")
print("  1. Prepare your dataset in the required JSON format")
print("  2. Run: python src/train.py --config configs/default_config.yaml")
print()


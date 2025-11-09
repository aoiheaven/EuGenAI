# 🚀 EuGenAI Training Commands Quick Reference

Complete command reference for training EuGenAI in different scenarios.

---

## 📋 Prerequisites

```bash
# 1. Setup environment on remote GPU
git clone https://github.com/yourusername/EuGenAI.git
cd EuGenAI

# 2. Create virtual environment
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# OR: .venv\Scripts\activate  # Windows

# 3. Install dependencies (GPU version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .

# 4. Verify GPU
python scripts/test_gpu.py
```

---

## 🎯 Training Modes Overview

| Mode | Data Required | Time | Cost | Accuracy | Use Case |
|------|--------------|------|------|----------|----------|
| **Self-Supervised** | Images + text (no labels) | 3 days | $115 | N/A | Pre-training from scratch |
| **Weak Supervision** | Images + diagnosis labels | 2 days | $40 | 85% | Limited annotations |
| **Reinforcement Learning** | Same as weak | 4 days | $80 | 88% | Improve CoT quality |
| **Full Supervision** | Images + diagnosis + CoT | 1 day | $25 | 92% | Best performance |

---

## 1️⃣ Self-Supervised Pre-training (No Labels)

**When to use**: You have lots of images but NO labels

### Data Format

```json
{
  "samples": [
    {
      "sample_id": "IMG_001",
      "image": {
        "path": "data/images/fundus_001.jpg",
        "modality": "fundus_photography"
      },
      "medical_record": {
        "age": 58,
        "gender": "M",
        "history": "Type 2 diabetes for 15 years..."
      }
    }
  ]
}
```

**No diagnosis, no CoT needed!**

### Commands

```bash
# Small scale test (1K images, ~8 hours, ~$10)
python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_unlabeled.json \
    --epochs 50 \
    --batch_size 32 \
    --output_dir checkpoints_ssl_1k

# Medium scale (5K images, ~2 days, ~$50)
python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_unlabeled.json \
    --epochs 100 \
    --batch_size 64 \
    --output_dir checkpoints_ssl_5k

# Full scale (10K+ images, ~3 days, ~$115)
python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_unlabeled.json \
    --epochs 100 \
    --batch_size 128 \
    --num_workers 8 \
    --output_dir checkpoints_ssl_full
```

**Output**: Pre-trained encoder weights for transfer learning

---

## 2️⃣ Weak Supervision (Diagnosis Labels Only)

**When to use**: You have images with diagnosis labels, but NO CoT annotations

### Data Format

```json
{
  "samples": [
    {
      "sample_id": "WS_001",
      "image": {
        "path": "data/images/fundus_001.jpg",
        "modality": "fundus_photography"
      },
      "medical_record": {
        "age": 58,
        "history": "Type 2 diabetes..."
      },
      "diagnosis": {
        "primary": "Severe NPDR",
        "confidence": 0.92
      }
    }
  ]
}
```

**Only diagnosis label needed! CoT auto-generated**

### Commands

```bash
# Option A: Train from scratch (not recommended)
python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --data_file data/train_weak_labels.json \
    --epochs 50 \
    --output_dir checkpoints_weak_scratch

# Option B: Start from SSL pre-trained model (recommended)
python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --pretrained checkpoints_ssl/best_model.pth \
    --data_file data/train_weak_labels.json \
    --epochs 30 \
    --output_dir checkpoints_weak

# With specific settings
python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --pretrained checkpoints_ssl/best_model.pth \
    --data_file data/train_weak_labels.json \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 50 \
    --output_dir checkpoints_weak_custom
```

**Expected**: 85% accuracy with auto-generated CoT

---

## 3️⃣ Reinforcement Learning (Improve CoT)

**When to use**: After weak supervision, to improve CoT quality

### Data Format

Same as weak supervision (diagnosis labels only)

### Commands

```bash
# Standard RL training
python src/train_reinforcement_learning.py \
    --config configs/reinforcement_learning_config.yaml \
    --pretrained checkpoints_weak/best_model.pth \
    --data_file data/train_weak_labels.json \
    --epochs 100 \
    --output_dir checkpoints_rl

# With custom reward weights
python src/train_reinforcement_learning.py \
    --config configs/reinforcement_learning_config.yaml \
    --pretrained checkpoints_weak/best_model.pth \
    --data_file data/train_weak_labels.json \
    --reward_accuracy 1.0 \
    --reward_attention 0.5 \
    --reward_coherence 0.3 \
    --reward_consistency 0.2 \
    --output_dir checkpoints_rl_custom
```

**Expected**: 88% accuracy with high-quality CoT

---

## 4️⃣ Full Supervision (Expert Annotations)

**When to use**: You have expert-annotated CoT data

### Data Format

```json
{
  "samples": [
    {
      "sample_id": "FULL_001",
      "image": {"path": "..."},
      "medical_record": {...},
      "chain_of_thought": {
        "reasoning_steps": [
          {
            "step": 1,
            "action": "Assess overall image quality",
            "observation": "Good quality fundus photo...",
            "region_of_interest": {
              "bbox": [512, 512, 1536, 1536],
              "description": "Central retina"
            }
          }
        ]
      },
      "final_diagnosis": {
        "primary": "Severe NPDR",
        "confidence": 0.92
      }
    }
  ]
}
```

### Commands

```bash
# Train from scratch (small expert dataset)
python src/train.py \
    --config configs/default_config.yaml \
    --data_file data/train_full_annotations.json \
    --epochs 50 \
    --output_dir checkpoints_full

# Fine-tune from SSL pre-trained (recommended)
python src/train.py \
    --config configs/default_config.yaml \
    --pretrained checkpoints_ssl/best_model.pth \
    --data_file data/train_full_annotations.json \
    --epochs 30 \
    --learning_rate 0.00005 \
    --output_dir checkpoints_full_finetuned

# Fine-tune from RL model (best)
python src/train.py \
    --config configs/default_config.yaml \
    --pretrained checkpoints_rl/best_model.pth \
    --data_file data/train_full_annotations.json \
    --epochs 20 \
    --learning_rate 0.00002 \
    --output_dir checkpoints_full_from_rl
```

**Expected**: 92% accuracy with expert-level CoT

---

## 🔄 Progressive Training Pipeline (Recommended)

**For maximum efficiency with minimal cost:**

```bash
# === STAGE 1: Self-Supervised Pre-training ===
# Use: 10,000 unlabeled images
# Time: 3 days on A100
# Cost: ~$115

python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_unlabeled_10k.json \
    --epochs 100 \
    --output_dir checkpoints_stage1_ssl

# === STAGE 2: Weak Supervision ===
# Use: 1,000 diagnosis labels ($5K to annotate)
# Time: 2 days on RTX 4090
# Cost: ~$40

python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --pretrained checkpoints_stage1_ssl/best_model.pth \
    --data_file data/train_diagnosis_1k.json \
    --epochs 50 \
    --output_dir checkpoints_stage2_weak

# === STAGE 3: Reinforcement Learning ===
# Use: Same 1,000 samples (no new data!)
# Time: 4 days on RTX 4090
# Cost: ~$80

python src/train_reinforcement_learning.py \
    --config configs/reinforcement_learning_config.yaml \
    --pretrained checkpoints_stage2_weak/best_model.pth \
    --data_file data/train_diagnosis_1k.json \
    --epochs 100 \
    --output_dir checkpoints_stage3_rl

# === STAGE 4: Optional Expert Fine-tuning ===
# Use: 200 expert CoT annotations ($2K)
# Time: 1 day
# Cost: ~$25

python src/train.py \
    --config configs/default_config.yaml \
    --pretrained checkpoints_stage3_rl/best_model.pth \
    --data_file data/train_expert_200.json \
    --epochs 20 \
    --learning_rate 0.00002 \
    --output_dir checkpoints_stage4_expert

# === RESULT ===
# Total Cost: $115 + $40 + $80 + $5K (labels) + $2K (optional) = ~$7,235
# vs Traditional: $50K (5,000 full annotations)
# Savings: 85%!
# Final Accuracy: 88% (without stage 4) or 92% (with stage 4)
```

---

## 💻 Hardware-Specific Commands

### For RTX 2060 (6GB)

```bash
python src/train.py \
    --config configs/rtx2060_config.yaml \
    --data_file data/train.json \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_workers 0
```

### For RTX 3090 / 4090 (24GB)

```bash
python src/train.py \
    --config configs/default_config.yaml \
    --data_file data/train.json \
    --batch_size 16 \
    --num_workers 8 \
    --mixed_precision true
```

### For A100 (40GB)

```bash
python src/train.py \
    --config configs/default_config.yaml \
    --data_file data/train.json \
    --batch_size 32 \
    --num_workers 16 \
    --mixed_precision true
```

---

## 📊 Monitoring Training

### Using TensorBoard

```bash
# In separate terminal
tensorboard --logdir logs/ --port 6006

# Access: http://localhost:6006
# Or remote: http://YOUR_SERVER_IP:6006
```

### Using WandB (Recommended)

```bash
# Setup (one time)
wandb login YOUR_API_KEY

# Enable in config or command line
python src/train.py \
    --config configs/default_config.yaml \
    --wandb_enabled \
    --wandb_project eugenai-training \
    --wandb_run_name experiment_001

# View at: https://wandb.ai/YOUR_USERNAME/eugenai-training
```

---

## 🔍 Resume Training

```bash
# Resume from last checkpoint
python src/train.py \
    --config configs/default_config.yaml \
    --resume checkpoints/checkpoint_epoch_10.pth

# Resume from specific checkpoint
python src/train.py \
    --config configs/default_config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --start_epoch 50
```

---

## 🧪 Evaluation

```bash
# Evaluate on test set
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --data_file data/test.json \
    --output_dir outputs/evaluation \
    --compute_metrics

# Generate predictions
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --data_file data/test.json \
    --output_dir outputs/predictions \
    --save_visualizations
```

---

## 🎯 Quick Decision Tree

```
Do you have ANY labels?
├─ NO → Use Self-Supervised (Stage 1)
│        Then get diagnosis labels
│        Then Weak Supervision (Stage 2)
│        Then RL (Stage 3)
│
└─ YES → Do you have CoT annotations?
         ├─ NO → Use Weak Supervision (Stage 2)
         │        Then RL (Stage 3)
         │
         └─ YES → Do you have pre-trained model?
                  ├─ YES → Fine-tune (best)
                  └─ NO → Full Supervision from scratch
```

---

## 💡 Best Practices

### 1. Always Start Small

```bash
# Test with 100 samples first
python src/train.py \
    --config configs/quick_test_config.yaml \
    --data_file data/test_100.json \
    --epochs 5

# Then scale up
python src/train.py \
    --config configs/default_config.yaml \
    --data_file data/train_full.json \
    --epochs 100
```

### 2. Use tmux/screen for Long Training

```bash
# Start tmux session
tmux new -s training

# Run training
python src/train.py --config configs/default_config.yaml

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### 3. Monitor GPU

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check memory
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

---

## 📞 Need Help?

- **Out of Memory**: Reduce `batch_size` or `image_size`
- **Training Slow**: Increase `num_workers` or use smaller model
- **Loss Not Decreasing**: Check data, reduce learning rate
- **Validation Worse**: Check for overfitting, add regularization

**For detailed troubleshooting, see**: [docs/REMOTE_TRAINING_GUIDE.md](REMOTE_TRAINING_GUIDE.md)

---

**Last Updated**: 2024-11-09  
**Version**: 1.0


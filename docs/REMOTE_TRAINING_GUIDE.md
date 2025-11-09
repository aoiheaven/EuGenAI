# 🚀 Remote GPU Training Guide for EuGenAI

Complete guide for training EuGenAI on remote GPU servers and cloud platforms.

---

## 📑 Table of Contents

1. [Quick Start Validation](#quick-start-validation)
2. [Recommended Platforms](#recommended-platforms)
3. [Cost Estimation](#cost-estimation)
4. [Setup Instructions](#setup-instructions)
5. [Remote Training Workflow](#remote-training-workflow)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Best Practices](#best-practices)

---

## Quick Start Validation

### Phase 1: Local Sanity Check (5 minutes)

Before spending money on cloud GPUs, verify everything works locally:

```bash
# 1. Create tiny test dataset
python scripts/create_quick_test_dataset.py \
    --num_train 20 \
    --num_val 5 \
    --num_test 5

# 2. Validate data format
python scripts/validate_data.py \
    --data_file data/quick_test_train.json \
    --mode weak_supervised

# 3. Run sanity check (CPU is fine)
python src/train.py \
    --config configs/quick_test_config.yaml \
    --sanity_check \
    --device cpu
```

**Expected result**: Should complete without errors in < 5 minutes.

---

### Phase 2: Cloud Quick Test (30-60 minutes, ~$1)

Once local sanity check passes, test on cloud GPU:

```bash
# 1. Create 100-sample dataset
python scripts/create_quick_test_dataset.py \
    --num_train 100 \
    --num_val 20 \
    --num_test 20

# 2. Upload to cloud (see setup instructions below)

# 3. Run quick training (10 epochs, ~30 min)
python src/train.py \
    --config configs/quick_test_config.yaml \
    --wandb_enabled
```

**Expected results**:
- Training completes successfully
- Loss decreases over epochs
- Validation accuracy > 60% (random is 20% for 5-class)
- Metrics logged to WandB/TensorBoard
- GPU memory usage < 6GB

**Cost**: $0.50-$1.00 (depending on platform)

---

### Phase 3: Small-Scale Full Test (8-12 hours, ~$20)

After quick test succeeds, scale up:

```bash
# Use 1000 samples, train for 50 epochs
python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_1k.json \
    --epochs 50
```

**Expected results**:
- Convergence on 1K samples
- Validation accuracy > 75%
- Confirm metrics and checkpointing work

---

## Recommended Platforms

### 🥇 Best for Beginners: Vast.ai

**Why**: Cheapest, easy to use, flexible

```bash
# Pricing (RTX 3090/4090)
GPU: RTX 3090 (24GB) - $0.20-0.40/hour
GPU: RTX 4090 (24GB) - $0.30-0.60/hour
GPU: A100 (40GB) - $1.00-1.50/hour

# Setup time: 5 minutes
# Min spend: ~$5
```

**Pros**:
- Pay-per-second billing
- No commitment
- Easy Docker setup
- Jupyter notebooks available

**Cons**:
- Instances can be interrupted
- Need to manage data uploads

**Best for**: Quick tests, experimentation

---

### 🥈 Best for Serious Training: Lambda Labs

**Why**: Reliable, good performance, fair pricing

```bash
# Pricing
GPU: RTX 6000 Ada (48GB) - $0.75/hour
GPU: A100 (40GB) - $1.10/hour
GPU: H100 (80GB) - $2.00/hour

# Setup time: 10 minutes
# Min spend: $50
```

**Pros**:
- Very stable
- Good documentation
- Persistent storage
- SSH access

**Cons**:
- Minimum $50 credit
- Less flexible than Vast.ai

**Best for**: Production training, serious projects

---

### 🥉 Most Features: Google Cloud / AWS / Azure

**Why**: Full ecosystem, integrations, managed services

```bash
# Pricing (GCP example)
GPU: T4 (16GB) - $0.35/hour
GPU: V100 (16GB) - $2.48/hour
GPU: A100 (40GB) - $3.67/hour

# Plus: Storage, networking, etc.
# Setup time: 30-60 minutes
```

**Pros**:
- Full MLOps ecosystem
- Excellent monitoring
- Auto-scaling
- Free credits for new users

**Cons**:
- Most expensive
- Complex setup
- Easy to overspend

**Best for**: Enterprise, production deployments

---

### 🎓 Academic Users: University Clusters

**Pricing**: Usually free!

**Pros**:
- Free GPU access
- Large-scale resources
- Support from IT staff

**Cons**:
- Job queues (wait times)
- May need approval
- Cluster-specific setup

---

## Cost Estimation

### Quick Test (100 samples, 10 epochs)
```
GPU: RTX 3090
Time: 30-60 minutes
Cost: $0.20/hr × 1 hr = $0.20-0.40
```

### Small-Scale Training (1K samples, 50 epochs)
```
GPU: RTX 4090
Time: 8-12 hours
Cost: $0.40/hr × 10 hrs = $4.00
```

### Self-Supervised Pre-training (10K samples, 100 epochs)
```
GPU: A100
Time: 3-5 days
Cost: $1.20/hr × 96 hrs = $115.20
```

### Full Pipeline (SSL → Weak → RL)
```
Phase 1 (SSL): $115 (3 days, A100)
Phase 2 (Weak): $40 (2 days, RTX 4090)
Phase 3 (RL): $80 (4 days, RTX 4090)
Total: ~$235 for complete training
```

**Cost savings vs. annotation**: 
- Traditional: $50,000 (5K full annotations)
- EuGenAI: $235 (training) + $5,000 (1K diagnosis labels) = **$5,235**
- **Savings: 89.5%** 🎉

---

## Setup Instructions

### Option 1: Vast.ai (Easiest)

```bash
# 1. Create account at vast.ai

# 2. Search for instance
# - GPU: RTX 3090 or RTX 4090
# - RAM: 32GB+
# - Storage: 100GB+
# - Image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 3. Rent instance and connect
ssh -p PORT root@INSTANCE_IP

# 4. Clone repo and setup
git clone https://github.com/aoiheaven/EuGenAI.git
cd EuGenAI
pip install -e .

# 5. Upload data
# (Use rsync or scp from local machine)
rsync -avz -e "ssh -p PORT" data/ root@INSTANCE_IP:/workspace/EuGenAI/data/

# 6. Start training
python src/train.py --config configs/quick_test_config.yaml
```

---

### Option 2: Lambda Labs

```bash
# 1. Create account and add credits

# 2. Launch instance
# - Select region
# - Choose GPU type
# - Select Ubuntu 20.04 + PyTorch

# 3. SSH connect
ssh ubuntu@INSTANCE_IP

# 4. Setup (same as Vast.ai above)
```

---

### Option 3: Google Colab Pro (Quick Testing)

```python
# In Colab notebook

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo
!git clone https://github.com/aoiheaven/EuGenAI.git
%cd EuGenAI

# 3. Install dependencies
!pip install -e .

# 4. Upload data to Google Drive, then:
!ln -s /content/drive/MyDrive/eugenai_data data

# 5. Train
!python src/train.py --config configs/quick_test_config.yaml
```

**Note**: Colab Pro has 24-hour limit. Good for quick tests only.

---

## Remote Training Workflow

### Step 1: Prepare Locally

```bash
# 1. Create and validate dataset
python scripts/create_quick_test_dataset.py
python scripts/validate_data.py --data_file data/quick_test_train.json

# 2. Test config locally (CPU sanity check)
python src/train.py --config configs/quick_test_config.yaml --sanity_check --device cpu

# 3. Compress data for upload
tar -czf eugenai_data.tar.gz data/
```

---

### Step 2: Setup Remote Instance

```bash
# On remote machine
git clone https://github.com/aoiheaven/EuGenAI.git
cd EuGenAI

# Install dependencies
pip install -e .

# Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

### Step 3: Upload Data

```bash
# From local machine
rsync -avz --progress data/ user@remote:/path/to/EuGenAI/data/

# Or use scp
scp -r data/ user@remote:/path/to/EuGenAI/data/
```

---

### Step 4: Start Training

```bash
# On remote machine

# Option A: Direct run (for short training)
python src/train.py --config configs/quick_test_config.yaml

# Option B: Use tmux/screen (recommended for long training)
tmux new -s training
python src/train.py --config configs/quick_test_config.yaml
# Press Ctrl+B, then D to detach
# Reconnect: tmux attach -t training

# Option C: Use nohup (background process)
nohup python src/train.py --config configs/quick_test_config.yaml > training.log 2>&1 &

# Monitor logs
tail -f training.log
```

---

### Step 5: Monitor Progress

```bash
# Option 1: WandB (recommended)
# - Set wandb_enabled: true in config
# - Login: wandb login YOUR_API_KEY
# - View at: https://wandb.ai/YOUR_USERNAME/eugenai-quick-test

# Option 2: TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
# Access: http://INSTANCE_IP:6006

# Option 3: Watch logs
tail -f logs/train.log

# Option 4: Check GPU usage
watch -n 1 nvidia-smi
```

---

### Step 6: Download Results

```bash
# From local machine

# Download checkpoints
rsync -avz user@remote:/path/to/EuGenAI/checkpoints_quick_test/ checkpoints/

# Download logs
rsync -avz user@remote:/path/to/EuGenAI/logs/ logs/

# Download outputs
rsync -avz user@remote:/path/to/EuGenAI/outputs/ outputs/
```

---

## Monitoring & Debugging

### Key Metrics to Watch

```python
# 1. Training Loss
# - Should decrease steadily
# - Target: < 0.5 for weak supervision

# 2. Validation Accuracy
# - Should increase over epochs
# - Quick test (100 samples): > 60%
# - Small scale (1K samples): > 75%
# - Full scale (10K+ samples): > 85%

# 3. GPU Utilization
# - Should be > 80%
# - If low, increase batch size or num_workers

# 4. Memory Usage
# - Monitor with nvidia-smi
# - Quick test: < 6GB
# - Full training: < 20GB
```

---

### Common Issues

#### ❌ Issue: CUDA Out of Memory

```yaml
# Solution: Reduce batch size in config
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Compensate
```

---

#### ❌ Issue: Training Too Slow

```yaml
# Solution: Optimize data loading
dataset:
  num_workers: 8  # Increase
  pin_memory: true
  prefetch_factor: 2
```

---

#### ❌ Issue: Loss Not Decreasing

```bash
# Check 1: Verify data quality
python scripts/validate_data.py --data_file data/train.json

# Check 2: Reduce learning rate
training:
  learning_rate: 1e-5  # Try lower

# Check 3: Check for bugs
python src/train.py --config configs/quick_test_config.yaml --debug
```

---

#### ❌ Issue: Instance Disconnected

```bash
# Solution: Use tmux or screen
tmux new -s training
python src/train.py ...
# Detach: Ctrl+B, D
# Reattach later: tmux attach -t training
```

---

## Best Practices

### ✅ Do's

1. **Always run sanity check first**
   - Test with 10-20 samples locally
   - Verify no errors before cloud training

2. **Use version control**
   ```bash
   git add configs/my_config.yaml
   git commit -m "Add training config"
   git push
   ```

3. **Monitor costs**
   - Set budget alerts on cloud platforms
   - Check instance pricing before starting
   - Stop instances when not in use

4. **Save checkpoints frequently**
   ```yaml
   training:
     save_every: 5  # Every 5 epochs
   ```

5. **Use remote monitoring**
   - WandB or TensorBoard
   - Can view progress from anywhere

6. **Backup important runs**
   ```bash
   # Regular backups to cloud storage
   rsync -avz checkpoints/ s3://my-bucket/eugenai-checkpoints/
   ```

---

### ❌ Don'ts

1. **Don't forget to stop instances**
   - Can rack up costs quickly
   - Set auto-shutdown if available

2. **Don't skip validation**
   - Always validate data before training
   - Saves debugging time

3. **Don't use absolute paths**
   - Use relative paths for portability

4. **Don't train without monitoring**
   - Always use WandB or TensorBoard
   - Catch issues early

5. **Don't skip quick tests**
   - Quick test (100 samples) before full training
   - Saves money and time

---

## Training Checklist

Before starting cloud training:

- [ ] Local sanity check passed (CPU)
- [ ] Data validated with validate_data.py
- [ ] Config tested with small dataset
- [ ] WandB/TensorBoard configured
- [ ] Checkpointing enabled
- [ ] Remote monitoring setup
- [ ] Budget alert set (if applicable)
- [ ] Estimated cost calculated
- [ ] tmux/screen session started
- [ ] Backup plan for checkpoints

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Monitor training
tail -f logs/train.log

# Check GPU usage live
watch -n 1 nvidia-smi

# Kill training
pkill -f train.py

# Check disk space
df -h

# Compress checkpoints
tar -czf checkpoints.tar.gz checkpoints_quick_test/

# Download to local
scp user@remote:/path/checkpoints.tar.gz .

# Resume training
python src/train.py --config configs/quick_test_config.yaml --resume checkpoints/last.pth
```

---

## Cost Optimization Tips

### 1. Use Spot Instances (50-80% cheaper)
```bash
# GCP Preemptible
# AWS Spot Instances
# Can save 50-80% but may be interrupted
```

### 2. Right-size GPU
```bash
# Quick test: RTX 3090 ($0.20/hr)
# Full training: RTX 4090 ($0.40/hr)
# Large-scale: A100 ($1.20/hr)
```

### 3. Mixed Precision Training
```yaml
training:
  mixed_precision: true  # FP16, saves memory
```

### 4. Efficient Data Loading
```yaml
dataset:
  num_workers: 8
  pin_memory: true
```

### 5. Gradient Checkpointing
```yaml
model:
  gradient_checkpointing: true  # Trades compute for memory
```

---

## Next Steps

After successful quick test:

1. **Scale up data**: 1K → 10K → 100K samples
2. **Run full pipeline**: SSL → Weak → RL
3. **Hyperparameter tuning**: Use WandB sweeps
4. **Production deployment**: Export to ONNX, optimize inference

---

## Support Resources

- **GitHub Issues**: Report bugs and ask questions
- **Documentation**: [Full docs](../README.md)
- **WandB Community**: Share runs and get feedback
- **Discord/Slack**: Join ML community servers

---

**Last Updated**: 2024-11-09  
**Version**: 1.0


# Quick Start Guide

Get started with the Medical Multimodal Chain-of-Thought framework in 5 minutes!

## Step 1: Setup Environment

```bash
# Run the automated setup script
bash setup.sh

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (if not already done)
uv pip install -e .
```

## Step 2: Prepare Your Data

### Option A: Use the example data format

```bash
# Check the example data format
cat data_format_example.json

# Create your own dataset following this format
# Place images in data/images/
# Create train.json, val.json, test.json in data/
```

### Option B: Use the preparation script

```bash
# Modify scripts/prepare_data.py for your data source
# Then run:
python scripts/prepare_data.py --input raw_data --output data
```

### Minimal data structure:

```json
{
  "dataset_info": {...},
  "samples": [
    {
      "sample_id": "MED_001",
      "image": {"path": "images/case1.jpg", ...},
      "medical_record": {"history": "...", ...},
      "chain_of_thought": {"reasoning_steps": [...]},
      "final_diagnosis": {"primary": "...", "confidence": 0.9}
    }
  ]
}
```

## Step 3: Configure Training

Edit `configs/default_config.yaml`:

```yaml
# Key settings to adjust
model:
  img_size: 512  # Your image size
  num_diagnosis_classes: 100  # Number of diagnosis types

dataset:
  train_file: "data/train.json"
  val_file: "data/val.json"

training:
  batch_size: 4  # Adjust based on your GPU memory
  num_epochs: 100
  learning_rate: 1.0e-4
```

## Step 4: Train the Model

```bash
# Start training
python src/train.py --config configs/default_config.yaml

# Monitor with TensorBoard (optional)
tensorboard --logdir logs
```

## Step 5: Run Inference

```bash
# Inference on a single case
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image data/images/test_case.jpg \
    --text "Patient is 65yo male with chest pain..." \
    --output outputs/test_case/

# Check outputs/test_case/ for:
# - attention_heatmap.png
# - chain_of_thought.png
# - report.json
```

## Example Workflow

```python
# In Python script or Jupyter notebook
from src.dataset import MedicalChainOfThoughtDataset
from src.model import MedicalMultimodalCoT
from src.inference import MedicalCoTInference

# 1. Load dataset
dataset = MedicalChainOfThoughtDataset(
    data_file='data/train.json',
    data_root='data',
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")

# 2. Initialize model
model = MedicalMultimodalCoT(
    img_size=512,
    num_diagnosis_classes=100,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Run inference (after training)
engine = MedicalCoTInference('checkpoints/best_model.pth')

results = engine.predict(
    image_path='data/images/case_001.jpg',
    clinical_text='Patient history...',
    reasoning_steps=[
        {'action': 'Step 1', 'observation': '...', 'bbox': [0, 0, 100, 100]}
    ]
)

print(f"Diagnosis confidence: {results['confidence']:.3f}")
print(f"Top diagnosis: {results['top_diagnoses'][0]}")
```

## Troubleshooting

### Out of GPU memory

```yaml
# In configs/default_config.yaml
training:
  batch_size: 2  # Reduce batch size
  
hardware:
  mixed_precision: true  # Enable if not already
```

### Dataset loading errors

```bash
# Validate your JSON format
python -m json.tool data/train.json > /dev/null
# Should not show errors if JSON is valid

# Check image paths
ls data/images/
```

### Model not learning

- Check if loss is decreasing in TensorBoard
- Verify data labels are correct
- Try reducing learning rate
- Ensure data augmentation is appropriate

## Next Steps

- Read the full documentation in [README.md](README.md)
- Explore model architecture in `src/model.py`
- Customize data loading in `src/dataset.py`
- Adjust training loop in `src/train.py`
- Create custom visualizations in `src/inference.py`

## Getting Help

- Check [GitHub Issues](https://github.com/aoiheaven/EuGenAI/issues)
- Read the detailed documentation in `README.md` or `README_zh.md`
- Contact: Via GitHub Issues

---

Happy training! 🚀


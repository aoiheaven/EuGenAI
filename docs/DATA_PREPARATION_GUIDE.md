# 📊 EuGenAI Data Preparation Guide

Complete guide to preparing your medical imaging data for training EuGenAI models.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Data Requirements by Training Mode](#data-requirements-by-training-mode)
3. [Step-by-Step Preparation](#step-by-step-preparation)
4. [Data Format Specifications](#data-format-specifications)
5. [Helper Scripts](#helper-scripts)
6. [Quality Control](#quality-control)
7. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

EuGenAI supports **4 training modes** with different data requirements:

| Mode | Images | Clinical Text | Diagnosis Labels | CoT Annotations | Segmentation | Cost/Sample |
|------|--------|---------------|------------------|-----------------|--------------|-------------|
| **Self-Supervised** | ✅ | ✅ | ❌ | ❌ | ❌ | $0 |
| **Weak Supervision** | ✅ | ✅ | ✅ | ❌ | ❌ | $5 |
| **Reinforcement Learning** | ✅ | ✅ | ✅ | ❌ | ❌ | $0 |
| **Full Supervision** | ✅ | ✅ | ✅ | ✅ | ✅ (optional) | $50 |

**Recommended Path**: Self-Supervised → Weak Supervision → RL → Optional Full Fine-tuning

---

## Data Requirements by Training Mode

### 🔵 Mode 1: Self-Supervised Learning (No Labels)

**What you need**:
- Medical images (JPG, PNG, DICOM)
- Basic clinical text (patient history, exam notes)
- **NO diagnosis labels needed**
- **NO CoT annotations needed**

**Minimum dataset size**: 5,000+ images (more is better)

**Example use case**: You have a large collection of fundus images from routine screenings but no diagnoses.

---

### 🟡 Mode 2: Weak Supervision (Diagnosis Only)

**What you need**:
- Medical images
- Clinical text
- **Diagnosis labels** (e.g., "Severe NPDR", "Intermediate AMD")
- **NO CoT annotations needed** (auto-generated)

**Minimum dataset size**: 500-1,000 labeled images

**Example use case**: You have images with final diagnoses from reports, but no detailed reasoning chains.

---

### 🟢 Mode 3: Reinforcement Learning (Same as Weak Supervision)

**What you need**: Same as Mode 2
- The model learns to generate better CoT through reward optimization
- No additional data required beyond weak supervision

---

### 🟣 Mode 4: Full Supervision (Complete Annotations)

**What you need**:
- Medical images
- Clinical text
- Diagnosis labels
- **Step-by-step Chain-of-Thought reasoning**
- Segmentation masks (optional but recommended)

**Minimum dataset size**: 200-500 fully annotated samples

**Example use case**: You have expert-annotated cases with detailed diagnostic reasoning.

---

## Step-by-Step Preparation

### 📁 Step 1: Organize Your Image Files

Create the following directory structure:

```
EuGenAI/
├── data/
│   ├── images/
│   │   ├── fundus_001.jpg
│   │   ├── fundus_002.jpg
│   │   ├── oct_001.png
│   │   └── ...
│   ├── train.json          # Training data metadata
│   ├── val.json            # Validation data metadata
│   └── test.json           # Test data metadata
```

**Image requirements**:
- **Format**: JPG, PNG, or DICOM (.dcm)
- **Resolution**: Minimum 512x512 (higher is better)
- **Color**: RGB for fundus photos, grayscale acceptable for OCT
- **Naming**: Use descriptive names (e.g., `patient123_OD_macula.jpg`)

---

### 📝 Step 2: Collect Clinical Information

For each image, gather:

1. **Patient Demographics** (anonymized):
   - Age
   - Gender
   - Relevant medical history

2. **Clinical History**:
   - Chief complaint
   - Duration of symptoms
   - Previous diagnoses

3. **Physical Examination**:
   - Visual acuity
   - Intraocular pressure
   - Slit lamp findings

4. **Laboratory Results** (if relevant):
   - HbA1c for diabetic patients
   - Blood pressure
   - Other relevant biomarkers

---

### 🏷️ Step 3: Prepare Labels Based on Training Mode

#### For Self-Supervised Learning (No Labels):

**You can skip this step!** Just prepare the JSON with images and clinical text.

**Example JSON entry**:

```json
{
  "sample_id": "SSL_001",
  "image": {
    "path": "images/fundus_001.jpg",
    "modality": "fundus_photography",
    "eye": "OD"
  },
  "medical_record": {
    "age": 58,
    "gender": "M",
    "history": "Type 2 diabetes for 15 years...",
    "physical_exam": "Visual acuity OD 20/40",
    "lab_results": "HbA1c: 8.2%"
  }
}
```

---

#### For Weak Supervision (Diagnosis Labels Only):

**What to label**: Final diagnosis only

**How to obtain**:
- Extract from existing medical reports
- Have ophthalmologists review images (5 minutes per case)
- Use crowdsourcing platforms (with medical professional verification)

**Example JSON entry**:

```json
{
  "sample_id": "WS_001",
  "image": {
    "path": "images/fundus_001.jpg",
    "modality": "fundus_photography",
    "eye": "OD"
  },
  "medical_record": {
    "age": 58,
    "gender": "M",
    "history": "Type 2 diabetes for 15 years...",
    "physical_exam": "Visual acuity OD 20/40",
    "lab_results": "HbA1c: 8.2%"
  },
  "diagnosis": {
    "primary": "Severe NPDR",
    "confidence": 0.92,
    "icd10": "E11.3491"
  }
}
```

**Diagnosis categories for ophthalmology**:
- Diabetic Retinopathy: No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR
- AMD: Normal, Early AMD, Intermediate AMD, Advanced AMD (GA/CNV)
- Glaucoma: Normal, Suspect, Mild, Moderate, Severe
- Others: Normal, Cataract, Macular hole, ERM, etc.

---

#### For Full Supervision (Complete CoT):

**What to label**: Everything from weak supervision PLUS step-by-step reasoning

**Expert time required**: 30-60 minutes per case

**Example JSON entry**:

```json
{
  "sample_id": "FULL_001",
  "image": {
    "path": "images/fundus_001.jpg",
    "modality": "fundus_photography",
    "eye": "OD"
  },
  "medical_record": {
    "age": 58,
    "gender": "M",
    "history": "Type 2 diabetes for 15 years...",
    "physical_exam": "Visual acuity OD 20/40",
    "lab_results": "HbA1c: 8.2%"
  },
  "chain_of_thought": {
    "reasoning_steps": [
      {
        "step": 1,
        "action": "Assess overall image quality",
        "observation": "Good quality fundus photo, macula and optic disc visible",
        "region_of_interest": {
          "bbox": [512, 512, 1536, 1536],
          "description": "Central retina"
        }
      },
      {
        "step": 2,
        "action": "Examine for diabetic lesions",
        "observation": "Multiple microaneurysms in all 4 quadrants, scattered dot hemorrhages",
        "region_of_interest": {
          "bbox": [800, 800, 1200, 1200],
          "description": "Posterior pole"
        },
        "reference_text": "Patient has poor glycemic control (HbA1c 8.2%)"
      },
      {
        "step": 3,
        "action": "Assess macular involvement",
        "observation": "Hard exudates present within 1 disc diameter of macula center",
        "region_of_interest": {
          "bbox": [900, 900, 1100, 1100],
          "description": "Macula"
        },
        "reasoning": "Exudates near macula suggest DME, explaining reduced visual acuity"
      },
      {
        "step": 4,
        "action": "Grade severity",
        "reasoning": "Severe NPDR: extensive hemorrhages/MAs, no neovascularization",
        "supporting_evidence": [
          "Multiple MAs in all quadrants",
          "Hard exudates present",
          "No NVD or NVE visible",
          "Meets 4-2-1 rule criteria"
        ]
      }
    ]
  },
  "final_diagnosis": {
    "primary": "Severe NPDR with DME",
    "secondary": ["Diabetic Macular Edema"],
    "confidence": 0.92,
    "urgency": "high",
    "recommendations": [
      "Refer for OCT to assess macular thickness",
      "Consider anti-VEGF injection",
      "Follow-up in 4-6 weeks"
    ]
  }
}
```

---

### 🔄 Step 4: Split Your Data

Create train/validation/test splits:

**Recommended ratios**:
- Training: 70%
- Validation: 15%
- Test: 15%

**Important considerations**:
- Split by **patient**, not by image (avoid data leakage)
- Ensure class balance across splits
- Keep test set untouched until final evaluation

---

### 💾 Step 5: Create JSON Files

Use our helper script to automate this process:

```bash
python scripts/prepare_data.py \
    --image_dir data/images/ \
    --output_dir data/ \
    --mode weak_supervised \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

Or manually create `train.json`, `val.json`, `test.json` following the format examples.

---

## Data Format Specifications

### Complete JSON Schema

See example files:
- `data_format_example.json` - Full supervision format
- `data_format_unlabeled_example.json` - Self-supervised format
- `data_format_weak_labels_example.json` - Weak supervision format

### Required Fields by Mode

#### All Modes:
```json
{
  "sample_id": "unique_identifier",
  "image": {
    "path": "relative/path/to/image.jpg",
    "modality": "fundus_photography|oct|angiography",
    "eye": "OD|OS"
  },
  "medical_record": {
    "age": 58,
    "gender": "M|F",
    "history": "text...",
    "physical_exam": "text...",
    "lab_results": "text..."
  }
}
```

#### Weak Supervision adds:
```json
{
  "diagnosis": {
    "primary": "diagnosis_name",
    "confidence": 0.92,
    "icd10": "code"
  }
}
```

#### Full Supervision adds:
```json
{
  "chain_of_thought": {
    "reasoning_steps": [...]
  },
  "final_diagnosis": {
    "primary": "diagnosis",
    "recommendations": [...]
  }
}
```

---

## Helper Scripts

### 1. Data Validation Script

```bash
# Validate your JSON files
python scripts/validate_data.py \
    --data_file data/train.json \
    --mode weak_supervised
```

Checks:
- JSON syntax validity
- Required fields present
- Image files exist
- Consistent data types

---

### 2. Data Statistics Script

```bash
# Get dataset statistics
python scripts/data_statistics.py \
    --data_file data/train.json
```

Outputs:
- Total samples
- Class distribution
- Image resolution statistics
- Missing field analysis

---

### 3. Data Augmentation Preview

```bash
# Preview augmentation effects
python scripts/preview_augmentation.py \
    --data_file data/train.json \
    --sample_ids SSL_001,SSL_002 \
    --output_dir preview/
```

---

### 4. Semi-Automated CoT Generation (Beta)

For users with diagnosis labels who want to bootstrap full annotations:

```bash
# Generate pseudo-CoT using GradCAM + GPT-4
python scripts/generate_pseudo_cot.py \
    --data_file data/train_weak.json \
    --model_checkpoint checkpoints/weak_model.pth \
    --llm_provider openai \
    --api_key YOUR_API_KEY \
    --output_file data/train_pseudo_cot.json
```

---

## Quality Control

### Checklist Before Training

- [ ] All image files exist and are readable
- [ ] No corrupt images (use validation script)
- [ ] JSON syntax is valid
- [ ] Train/val/test have no patient overlap
- [ ] Class distribution is reasonable (not 99% one class)
- [ ] Clinical text is anonymized (no PHI)
- [ ] Diagnosis labels are consistent (check spelling)
- [ ] Image resolutions are sufficient (>512x512)
- [ ] For CoT: Reasoning steps reference correct image regions

### Recommended Quality Metrics

**For Weak Supervision**:
- Inter-annotator agreement: >80% (Cohen's Kappa >0.75)
- Have 10% of cases double-checked by senior expert

**For Full Supervision**:
- Expert review time: 30-60 min per case
- CoT steps should be 3-6 per case
- Each step should reference specific image regions
- Have 20% of annotations peer-reviewed

---

## Common Issues & Solutions

### ❌ Issue: "File not found" error

**Cause**: Image paths in JSON don't match actual file locations

**Solution**:
```bash
# Check all image paths
python scripts/validate_data.py --data_file data/train.json --check_images
```

Fix paths to be relative to the EuGenAI root directory:
- ✅ Good: `"path": "data/images/fundus_001.jpg"`
- ❌ Bad: `"path": "/home/user/images/fundus_001.jpg"`

---

### ❌ Issue: Class imbalance (99% normal cases)

**Cause**: Dataset doesn't represent real clinical distribution

**Solution**:
- Oversample minority classes during training (handled automatically)
- Collect more pathological cases
- Use class weights in loss function (see config)

---

### ❌ Issue: Low model performance despite labels

**Cause**: Inconsistent or noisy labels

**Solution**:
1. Review label quality (spot-check 50 random samples)
2. Check inter-annotator agreement
3. Use confidence thresholding (remove low-confidence labels)
4. Start with self-supervised pre-training to build robust features

---

### ❌ Issue: Out of memory during training

**Cause**: Images too large or batch size too high

**Solution**:
```yaml
# In config YAML
data:
  image_size: 512  # Reduce from 1024
  
training:
  batch_size: 8    # Reduce from 32
  gradient_accumulation_steps: 4  # Simulate larger batch
```

---

## Cost Estimation

### Annotation Costs (approximate)

| Task | Time/Case | Cost/Case (US) | 1000 Cases |
|------|-----------|----------------|------------|
| Diagnosis label only | 5 min | $5 | $5,000 |
| CoT annotation | 45 min | $45 | $45,000 |
| Segmentation mask | 30 min | $30 | $30,000 |
| Full annotation | 90 min | $80 | $80,000 |

**Cost saving with progressive training**:
- Traditional: 1000 full annotations = $80,000
- EuGenAI approach: 1000 diagnosis labels = $5,000
- **Savings: 94%** 🎉

---

## Example Workflows

### Workflow 1: Starting from Scratch

```bash
# Step 1: Collect unlabeled images (free)
# - 10,000 fundus images from routine screenings
# - Basic clinical notes from EMR

# Step 2: Self-supervised pre-training (3 days)
python src/train_self_supervised.py \
    --config configs/self_supervised_config.yaml \
    --data_file data/train_unlabeled.json

# Step 3: Label 1000 cases with diagnosis only ($5K on MTurk + ophth verification)
# - Use crowdsourcing for initial labels
# - Have board-certified ophthalmologist verify

# Step 4: Weak supervision (2 days)
python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --pretrained checkpoints_ssl/best_model.pth

# Step 5: Reinforcement learning (4 days)
python src/train_reinforcement_learning.py \
    --config configs/reinforcement_learning_config.yaml \
    --pretrained checkpoints_weak/best_model.pth

# Result: 88% accuracy, $5K total cost
```

---

### Workflow 2: You Have Diagnosis Labels

```bash
# Skip Step 1-2, start from Step 3

# Step 3: Weak supervision with existing labels (2 days)
python src/train_weak_supervised.py \
    --config configs/weak_supervised_config.yaml \
    --data_file data/train_with_diagnoses.json

# Step 4: RL to improve CoT quality (4 days)
python src/train_reinforcement_learning.py \
    --config configs/reinforcement_learning_config.yaml \
    --pretrained checkpoints_weak/best_model.pth

# Result: 88% accuracy, no additional annotation cost!
```

---

### Workflow 3: You Have Some Expert CoT

```bash
# You have 200 expert-annotated cases

# Step 1: Pre-train on unlabeled data
python src/train_self_supervised.py --config configs/self_supervised_config.yaml

# Step 2: Fine-tune on 200 expert cases
python src/train.py \
    --config configs/default_config.yaml \
    --pretrained checkpoints_ssl/best_model.pth \
    --data_file data/train_expert_200.json

# Result: 92% accuracy with minimal expert time
```

---

## Best Practices

### ✅ Do's

1. **Start with self-supervised learning** if you have unlabeled data
2. **Anonymize all patient data** before creating JSON files
3. **Split by patient ID**, not randomly by images
4. **Validate data** before starting training
5. **Keep original data** - don't overwrite raw files
6. **Document your annotation process** for reproducibility
7. **Use consistent terminology** for diagnoses

### ❌ Don'ts

1. **Don't mix different image modalities** in the same training run
2. **Don't skip data validation** - it catches 90% of issues
3. **Don't use absolute file paths** - use relative paths
4. **Don't forget to hold out test data** - it should never be seen during training
5. **Don't over-compress images** - keep quality high (>90% JPEG quality)

---

## Need Help?

- **Example datasets**: Check `data_format_*_example.json` files
- **Issues**: [GitHub Issues](https://github.com/aoiheaven/EuGenAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aoiheaven/EuGenAI/discussions)
- **Email**: Contact via GitHub

---

## Additional Resources

- [Training Pipeline Guide](training_pipeline.md)
- [SSL & RL Implementation Details](SSL_RL_IMPLEMENTATION.md)
- [Quick Start Guide](../QUICKSTART.md)
- [Full Documentation](../README.md)

---

**Last Updated**: 2025-11-09  
**Version**: 1.0


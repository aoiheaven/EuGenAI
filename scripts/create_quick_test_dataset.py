#!/usr/bin/env python3
"""
Create a small test dataset for quick validation

This script creates a minimal dataset to verify:
1. Data loading works
2. Model can train
3. Metrics are computed correctly
4. Pipeline is functional

Usage:
    python scripts/create_quick_test_dataset.py --source eyepacs --output data/
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
import shutil


def create_mock_dataset(num_train: int = 100, num_val: int = 20, num_test: int = 20):
    """
    Create a mock dataset for testing
    
    Args:
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
    """
    
    dr_levels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    
    def create_sample(idx: int, split: str) -> Dict:
        """Create a single mock sample"""
        level = random.randint(0, 4)
        
        return {
            "sample_id": f"{split.upper()}_{idx:04d}",
            "image": {
                "path": f"data/images/quick_test/{split}_{idx:04d}.jpg",
                "modality": "fundus_photography",
                "eye": "OD" if idx % 2 == 0 else "OS",
                "resolution": [224, 224]
            },
            "medical_record": {
                "age": random.randint(40, 80),
                "gender": "M" if idx % 2 == 0 else "F",
                "history": f"Patient {idx}: Type 2 diabetes mellitus for {random.randint(5, 20)} years. "
                          f"Vision changes reported over past {random.randint(1, 12)} months.",
                "physical_exam": f"Visual acuity {['20/20', '20/30', '20/40', '20/60', '20/100'][level]}. "
                                f"Intraocular pressure: {random.randint(12, 21)} mmHg.",
                "lab_results": f"HbA1c: {random.uniform(5.5, 10.5):.1f}%, "
                              f"Fasting glucose: {random.randint(90, 200)} mg/dL"
            },
            "diagnosis": {
                "primary": dr_levels[level],
                "confidence": random.uniform(0.85, 0.99),
                "icd10": f"E11.349{level}"
            }
        }
    
    # Create datasets
    train_data = {
        "dataset_info": {
            "name": "quick_test_train",
            "version": "1.0",
            "description": "Quick test dataset for validation",
            "total_samples": num_train
        },
        "samples": [create_sample(i, "train") for i in range(num_train)]
    }
    
    val_data = {
        "dataset_info": {
            "name": "quick_test_val",
            "version": "1.0",
            "description": "Quick test validation dataset",
            "total_samples": num_val
        },
        "samples": [create_sample(i, "val") for i in range(num_val)]
    }
    
    test_data = {
        "dataset_info": {
            "name": "quick_test_test",
            "version": "1.0",
            "description": "Quick test evaluation dataset",
            "total_samples": num_test
        },
        "samples": [create_sample(i, "test") for i in range(num_test)]
    }
    
    return train_data, val_data, test_data


def save_dataset(data: Dict, output_file: Path):
    """Save dataset to JSON file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {data['dataset_info']['total_samples']} samples to {output_file}")


def download_sample_images(num_samples: int = 20):
    """
    Download a few sample images from public datasets
    This is a placeholder - implement actual download logic
    """
    print("\n📥 Image Download Instructions:")
    print("=" * 60)
    print("To get real images for testing, you have two options:")
    print("\n1. Download from Kaggle (Recommended):")
    print("   kaggle competitions download -c diabetic-retinopathy-detection")
    print("   unzip diabetic-retinopathy-detection.zip")
    print(f"   # Move {num_samples} images to data/images/quick_test/")
    print("\n2. Use synthetic images (for pipeline testing only):")
    print("   python scripts/generate_synthetic_fundus.py \\")
    print(f"       --num_images {num_samples} \\")
    print("       --output_dir data/images/quick_test/")
    print("\n3. Use your own images:")
    print("   # Place your fundus images in data/images/quick_test/")
    print("   # Update image paths in the JSON files accordingly")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create quick test dataset for EuGenAI validation"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=20,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=20,
        help="Number of test samples"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output directory for JSON files"
    )
    
    args = parser.parse_args()
    
    print("🚀 Creating Quick Test Dataset")
    print("=" * 60)
    
    # Create datasets
    train_data, val_data, test_data = create_mock_dataset(
        args.num_train,
        args.num_val,
        args.num_test
    )
    
    # Save datasets
    save_dataset(train_data, args.output_dir / "quick_test_train.json")
    save_dataset(val_data, args.output_dir / "quick_test_val.json")
    save_dataset(test_data, args.output_dir / "quick_test_test.json")
    
    # Image instructions
    total_images = args.num_train + args.num_val + args.num_test
    download_sample_images(total_images)
    
    print("\n✅ Dataset creation complete!")
    print("\nNext steps:")
    print("1. Obtain actual fundus images (see instructions above)")
    print("2. Validate dataset: python scripts/validate_data.py --data_file data/quick_test_train.json")
    print("3. Run quick test: python src/train.py --config configs/quick_test_config.yaml")


if __name__ == "__main__":
    main()


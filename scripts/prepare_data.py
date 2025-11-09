#!/usr/bin/env python3
"""
Data Preparation Script

Helper script to convert your raw medical data into the required JSON format.
Modify this script according to your specific data source.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def create_sample_template() -> Dict:
    """Create a template for a single data sample"""
    return {
        "sample_id": "MED_XXX",
        "patient_info": {
            "age": 0,
            "gender": "M/F",
            "chief_complaint": "Main symptom or complaint"
        },
        "image": {
            "path": "images/sample.jpg",
            "modality": "CT/MRI/Xray/Ultrasound",
            "body_part": "chest/abdomen/brain/etc",
            "metadata": {
                "resolution": [512, 512],
                "acquisition_date": "YYYY-MM-DD"
            }
        },
        "medical_record": {
            "history": "Patient medical history...",
            "physical_exam": "Physical examination findings...",
            "lab_results": "Laboratory test results..."
        },
        "chain_of_thought": {
            "reasoning_steps": [
                {
                    "step": 1,
                    "action": "Action description",
                    "observation": "What is observed",
                    "region_of_interest": {
                        "bbox": [x1, y1, x2, y2],
                        "description": "Region description"
                    },
                    "reasoning": "Reasoning for this step (optional)"
                }
            ],
            "intermediate_conclusions": [
                "Intermediate finding 1",
                "Intermediate finding 2"
            ]
        },
        "final_diagnosis": {
            "primary": "Primary diagnosis",
            "secondary": ["Secondary diagnosis 1"],
            "confidence": 0.0,
            "urgency": "urgent/semi-urgent/routine",
            "recommendations": ["Recommendation 1"]
        },
        "ground_truth": {
            "diagnosis": "Verified diagnosis",
            "verified_by": "Doctor name",
            "verification_date": "YYYY-MM-DD"
        }
    }


def validate_sample(sample: Dict) -> bool:
    """
    Validate that a sample has all required fields
    
    Args:
        sample: Sample dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'sample_id',
        'image',
        'medical_record',
        'chain_of_thought',
        'final_diagnosis'
    ]
    
    for field in required_fields:
        if field not in sample:
            print(f"Warning: Sample {sample.get('sample_id', 'unknown')} missing field: {field}")
            return False
    
    return True


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple:
    """
    Split dataset into train/val/test sets
    
    Args:
        samples: List of all samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    return train_samples, val_samples, test_samples


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Main function to prepare dataset
    
    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed JSON files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement your data loading logic here
    # This is a placeholder - you need to adapt it to your data format
    
    print("Loading raw data...")
    samples = []
    
    # Example: Load from your custom format
    # for file in input_path.glob("*.json"):
    #     with open(file) as f:
    #         data = json.load(f)
    #         sample = convert_to_format(data)  # Your conversion function
    #         if validate_sample(sample):
    #             samples.append(sample)
    
    # For demonstration, create a few sample entries
    print("Creating sample data...")
    for i in range(10):
        sample = create_sample_template()
        sample['sample_id'] = f"MED_{i:03d}"
        sample['image']['path'] = f"images/sample_{i:03d}.jpg"
        samples.append(sample)
    
    print(f"Total samples: {len(samples)}")
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        samples, train_ratio, val_ratio, test_ratio
    )
    
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    
    # Save datasets
    datasets = {
        'train.json': train_samples,
        'val.json': val_samples,
        'test.json': test_samples,
    }
    
    for filename, data in datasets.items():
        output_file = output_path / filename
        dataset_dict = {
            "dataset_info": {
                "name": "Medical Multimodal Chain-of-Thought Dataset",
                "version": "1.0",
                "description": "Medical imaging with chain-of-thought reasoning"
            },
            "samples": data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file}")
    
    print("\nDataset preparation complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare medical dataset")
    parser.add_argument(
        '--input',
        type=str,
        default='raw_data',
        help='Input directory with raw data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for processed JSON files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Data Validation Script for EuGenAI

Validates JSON data files to ensure they meet the requirements for training.

Usage:
    python scripts/validate_data.py --data_file data/train.json --mode weak_supervised
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


class DataValidator:
    """Validator for EuGenAI dataset JSON files"""
    
    def __init__(self, mode: str = 'weak_supervised'):
        self.mode = mode
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'missing_images': 0,
            'missing_fields': 0,
        }
    
    def validate_file(self, data_file: Path, check_images: bool = True) -> bool:
        """
        Validate a JSON data file
        
        Args:
            data_file: Path to JSON file
            check_images: Whether to check if image files exist
            
        Returns:
            True if validation passes, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Validating: {data_file}")
        print(f"Mode: {self.mode}")
        print(f"{'='*60}\n")
        
        # Check file exists
        if not data_file.exists():
            self.errors.append(f"File not found: {data_file}")
            return False
        
        # Load JSON
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON syntax: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False
        
        # Get samples
        if isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
        elif isinstance(data, list):
            samples = data
        else:
            self.errors.append("Invalid data structure. Expected 'samples' list or array of samples.")
            return False
        
        self.stats['total_samples'] = len(samples)
        
        # Validate each sample
        for idx, sample in enumerate(samples):
            self._validate_sample(sample, idx, data_file.parent, check_images)
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_sample(self, sample: Dict, idx: int, base_path: Path, check_images: bool):
        """Validate a single sample"""
        sample_id = sample.get('sample_id', f'index_{idx}')
        
        # Required fields for all modes
        required_fields = ['sample_id', 'image', 'medical_record']
        
        # Additional required fields by mode
        if self.mode in ['weak_supervised', 'reinforcement_learning']:
            required_fields.append('diagnosis')
        elif self.mode == 'full_supervised':
            required_fields.extend(['diagnosis', 'chain_of_thought', 'final_diagnosis'])
        
        # Check required fields
        for field in required_fields:
            if field not in sample:
                self.errors.append(f"Sample {sample_id}: Missing required field '{field}'")
                self.stats['missing_fields'] += 1
                return
        
        # Validate image field
        if not self._validate_image(sample['image'], sample_id, base_path, check_images):
            return
        
        # Validate medical_record
        if not self._validate_medical_record(sample['medical_record'], sample_id):
            return
        
        # Validate diagnosis if present
        if 'diagnosis' in sample:
            if not self._validate_diagnosis(sample['diagnosis'], sample_id):
                return
        
        # Validate chain_of_thought if present
        if 'chain_of_thought' in sample:
            if not self._validate_cot(sample['chain_of_thought'], sample_id):
                return
        
        self.stats['valid_samples'] += 1
    
    def _validate_image(self, image: Dict, sample_id: str, base_path: Path, check_files: bool) -> bool:
        """Validate image field"""
        required = ['path', 'modality']
        for field in required:
            if field not in image:
                self.errors.append(f"Sample {sample_id}: Image missing '{field}'")
                return False
        
        # Check modality
        valid_modalities = [
            'fundus_photography', 'oct', 'angiography', 'fa', 'octa',
            'CT', 'MRI', 'X-ray', 'ultrasound', 'other'
        ]
        if image['modality'] not in valid_modalities:
            self.warnings.append(
                f"Sample {sample_id}: Unknown modality '{image['modality']}'. "
                f"Valid: {', '.join(valid_modalities)}"
            )
        
        # Check if image file exists
        if check_files:
            image_path = base_path / image['path']
            if not image_path.exists():
                self.errors.append(f"Sample {sample_id}: Image file not found: {image['path']}")
                self.stats['missing_images'] += 1
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz']
            if not any(str(image_path).lower().endswith(ext) for ext in valid_extensions):
                self.warnings.append(
                    f"Sample {sample_id}: Unusual image extension. "
                    f"Valid: {', '.join(valid_extensions)}"
                )
        
        return True
    
    def _validate_medical_record(self, medical_record: Dict, sample_id: str) -> bool:
        """Validate medical_record field"""
        # At least one of these should be present
        optional_fields = ['age', 'gender', 'history', 'physical_exam', 'lab_results']
        
        has_any = any(field in medical_record for field in optional_fields)
        if not has_any:
            self.warnings.append(
                f"Sample {sample_id}: Medical record has no recognized fields. "
                f"Expected at least one of: {', '.join(optional_fields)}"
            )
        
        # Validate age if present
        if 'age' in medical_record:
            age = medical_record['age']
            if not isinstance(age, (int, float)) or age < 0 or age > 120:
                self.warnings.append(f"Sample {sample_id}: Unusual age value: {age}")
        
        # Validate gender if present
        if 'gender' in medical_record:
            gender = medical_record['gender']
            if gender not in ['M', 'F', 'Male', 'Female', 'Other', 'Unknown']:
                self.warnings.append(f"Sample {sample_id}: Unusual gender value: {gender}")
        
        return True
    
    def _validate_diagnosis(self, diagnosis: Dict, sample_id: str) -> bool:
        """Validate diagnosis field"""
        if 'primary' not in diagnosis:
            self.errors.append(f"Sample {sample_id}: Diagnosis missing 'primary' field")
            return False
        
        # Check confidence if present
        if 'confidence' in diagnosis:
            conf = diagnosis['confidence']
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                self.warnings.append(
                    f"Sample {sample_id}: Confidence should be between 0 and 1, got: {conf}"
                )
        
        return True
    
    def _validate_cot(self, cot: Dict, sample_id: str) -> bool:
        """Validate chain_of_thought field"""
        if 'reasoning_steps' not in cot:
            self.errors.append(f"Sample {sample_id}: CoT missing 'reasoning_steps'")
            return False
        
        steps = cot['reasoning_steps']
        if not isinstance(steps, list) or len(steps) == 0:
            self.errors.append(f"Sample {sample_id}: CoT reasoning_steps should be non-empty list")
            return False
        
        # Validate each step
        for step_idx, step in enumerate(steps):
            if 'step' not in step:
                self.warnings.append(f"Sample {sample_id}: Step {step_idx} missing 'step' number")
            
            if 'observation' not in step and 'reasoning' not in step:
                self.warnings.append(
                    f"Sample {sample_id}: Step {step_idx} missing 'observation' or 'reasoning'"
                )
            
            # Check region_of_interest if present
            if 'region_of_interest' in step:
                roi = step['region_of_interest']
                if 'bbox' in roi:
                    bbox = roi['bbox']
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        self.warnings.append(
                            f"Sample {sample_id}: Step {step_idx} bbox should be [x1, y1, x2, y2]"
                        )
        
        return True
    
    def _print_results(self):
        """Print validation results"""
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Total Samples:  {self.stats['total_samples']}")
        print(f"Valid Samples:  {self.stats['valid_samples']}")
        print(f"Missing Images: {self.stats['missing_images']}")
        print(f"Missing Fields: {self.stats['missing_fields']}")
        print()
        
        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"   - {error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more")
            print()
        
        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"   - {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")
            print()
        
        if not self.errors and not self.warnings:
            print("✅ All checks passed!")
        elif not self.errors:
            print("✅ Validation passed with warnings")
        else:
            print("❌ Validation failed")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate EuGenAI dataset JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate self-supervised data
    python scripts/validate_data.py --data_file data/train_unlabeled.json --mode self_supervised
    
    # Validate weak supervision data with image file checks
    python scripts/validate_data.py --data_file data/train.json --mode weak_supervised --check_images
    
    # Validate full supervision data
    python scripts/validate_data.py --data_file data/train.json --mode full_supervised

Modes:
    self_supervised      - Only images and clinical text required
    weak_supervised      - Images, clinical text, and diagnosis labels
    reinforcement_learning - Same as weak_supervised
    full_supervised      - Complete annotations including CoT
        """
    )
    
    parser.add_argument(
        '--data_file',
        type=Path,
        required=True,
        help='Path to JSON data file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['self_supervised', 'weak_supervised', 'reinforcement_learning', 'full_supervised'],
        default='weak_supervised',
        help='Training mode (determines required fields)'
    )
    
    parser.add_argument(
        '--check_images',
        action='store_true',
        help='Check if image files actually exist'
    )
    
    parser.add_argument(
        '--no_check_images',
        dest='check_images',
        action='store_false',
        help='Skip checking if image files exist'
    )
    
    parser.set_defaults(check_images=True)
    
    args = parser.parse_args()
    
    # Validate
    validator = DataValidator(mode=args.mode)
    success = validator.validate_file(args.data_file, check_images=args.check_images)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


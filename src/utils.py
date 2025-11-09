"""
Utility Functions for Medical Multimodal Chain-of-Thought Framework

Includes:
- Diagnosis label encoding/decoding
- Text processing utilities
- Validation helpers
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer


class DiagnosisLabelEncoder:
    """
    Encode diagnosis text to class labels and vice versa
    
    Handles both single-label and multi-label scenarios.
    Creates label vocabulary automatically from dataset.
    """
    
    def __init__(self, label_file: Optional[str] = None):
        """
        Args:
            label_file: Path to JSON file with label mappings
                       If None, will build vocabulary dynamically
        """
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.num_classes = 0
        
        if label_file and os.path.exists(label_file):
            self.load_from_file(label_file)
    
    def load_from_file(self, file_path: str):
        """Load label mappings from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.label_to_id = data.get('label_to_id', {})
            self.id_to_label = {int(v): k for k, v in self.label_to_id.items()}
            self.num_classes = len(self.label_to_id)
    
    def save_to_file(self, file_path: str):
        """Save label mappings to JSON file"""
        data = {
            'label_to_id': self.label_to_id,
            'id_to_label': {int(k): v for k, v in self.id_to_label.items()},
            'num_classes': self.num_classes
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_label(self, label: str) -> int:
        """
        Add a new label to vocabulary
        
        Args:
            label: Diagnosis label text
            
        Returns:
            Label ID
        """
        if label not in self.label_to_id:
            label_id = self.num_classes
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
            self.num_classes += 1
            return label_id
        return self.label_to_id[label]
    
    def build_from_dataset(self, dataset_file: str, save_path: Optional[str] = None):
        """
        Build label vocabulary from dataset
        
        Args:
            dataset_file: Path to dataset JSON file
            save_path: Path to save label mappings (optional)
        """
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        
        # Collect all unique diagnoses
        for sample in samples:
            diagnosis = sample.get('final_diagnosis', {})
            primary = diagnosis.get('primary', 'Unknown')
            self.add_label(primary)
            
            # Also add secondary diagnoses
            for secondary in diagnosis.get('secondary', []):
                self.add_label(secondary)
        
        print(f"Built vocabulary with {self.num_classes} diagnosis labels")
        
        if save_path:
            self.save_to_file(save_path)
    
    def encode(self, diagnosis_text: str) -> int:
        """
        Encode diagnosis text to label ID
        
        Args:
            diagnosis_text: Diagnosis string (can contain multiple diagnoses)
            
        Returns:
            Label ID (uses first/primary diagnosis)
        """
        # Extract primary diagnosis (before first comma)
        primary = diagnosis_text.split(',')[0].strip()
        
        # Return ID or 0 (unknown) if not in vocabulary
        return self.label_to_id.get(primary, 0)
    
    def decode(self, label_id: int) -> str:
        """
        Decode label ID to diagnosis text
        
        Args:
            label_id: Label ID
            
        Returns:
            Diagnosis text
        """
        return self.id_to_label.get(label_id, "Unknown")
    
    def encode_batch(self, diagnosis_texts: List[str]) -> torch.Tensor:
        """
        Encode batch of diagnosis texts
        
        Args:
            diagnosis_texts: List of diagnosis strings
            
        Returns:
            Tensor of label IDs [batch_size]
        """
        label_ids = [self.encode(text) for text in diagnosis_texts]
        return torch.tensor(label_ids, dtype=torch.long)


class TextProcessor:
    """
    Unified text processing for clinical text and reasoning steps
    """
    
    def __init__(self, tokenizer_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def encode_clinical_text(
        self,
        history: str,
        physical_exam: str,
        lab_results: str
    ) -> Dict[str, torch.Tensor]:
        """
        Encode clinical text components
        
        Args:
            history: Patient history
            physical_exam: Physical examination findings
            lab_results: Laboratory results
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Combine all clinical information
        full_text = f"History: {history} Physical Exam: {physical_exam} Lab Results: {lab_results}"
        
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def encode_reasoning_step(self, step: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode a single reasoning step
        
        Args:
            step: Dictionary with 'action', 'observation', 'reasoning' keys
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        action = step.get('action', '')
        observation = step.get('observation', '')
        reasoning = step.get('reasoning', '')
        
        # Construct step text
        step_text = f"Action: {action}. Observation: {observation}."
        if reasoning:
            step_text += f" Reasoning: {reasoning}"
        
        encoded = self.tokenizer(
            step_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def encode_reasoning_steps(
        self,
        steps: List[Dict],
        max_steps: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple reasoning steps
        
        Args:
            steps: List of step dictionaries
            max_steps: Maximum number of steps to encode
            
        Returns:
            Dictionary with stacked input_ids and attention_masks
        """
        encoded_steps = []
        
        for step in steps[:max_steps]:
            encoded = self.encode_reasoning_step(step)
            encoded_steps.append(encoded)
        
        # Pad to max_steps with empty encodings
        while len(encoded_steps) < max_steps:
            empty_encoded = {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }
            encoded_steps.append(empty_encoded)
        
        # Stack into tensors
        return {
            'input_ids': torch.stack([s['input_ids'] for s in encoded_steps]),
            'attention_mask': torch.stack([s['attention_mask'] for s in encoded_steps]),
            'num_steps': min(len(steps), max_steps)
        }


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = {
        'model': ['image_encoder_name', 'text_encoder_name', 'hidden_dim'],
        'dataset': ['data_root', 'train_file', 'val_file'],
        'training': ['batch_size', 'num_epochs', 'learning_rate']
    }
    
    errors = []
    
    # Check required sections
    for section, fields in required_fields.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue
        
        # Check required fields in section
        for field in fields:
            if field not in config[section]:
                errors.append(f"Missing required field: {section}.{field}")
    
    # Validate value ranges
    if 'training' in config:
        training = config['training']
        
        if 'learning_rate' in training and training['learning_rate'] <= 0:
            errors.append("Learning rate must be positive")
        
        if 'batch_size' in training and training['batch_size'] < 1:
            errors.append("Batch size must be >= 1")
        
        if 'num_epochs' in training and training['num_epochs'] < 1:
            errors.append("Number of epochs must be >= 1")
    
    if errors:
        raise ValueError(
            f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return True


def ensure_directories(config: Dict):
    """
    Create necessary directories based on config
    
    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config.get('logging', {}).get('log_dir', 'logs'),
        config.get('checkpoint', {}).get('save_dir', 'checkpoints'),
        config.get('dataset', {}).get('data_root', 'data'),
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")


def compute_class_weights(dataset, label_encoder: DiagnosisLabelEncoder) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        dataset: Dataset object
        label_encoder: Label encoder with vocabulary
        
    Returns:
        Class weights tensor [num_classes]
    """
    from collections import Counter
    
    # Count label frequencies
    label_counts = Counter()
    
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        diagnosis = sample.get('final_diagnosis', {}).get('primary', 'Unknown')
        label_id = label_encoder.encode(diagnosis)
        label_counts[label_id] += 1
    
    # Compute weights (inverse frequency)
    num_classes = label_encoder.num_classes
    weights = torch.ones(num_classes)
    
    total_samples = sum(label_counts.values())
    
    for label_id, count in label_counts.items():
        if count > 0:
            weights[label_id] = total_samples / (num_classes * count)
    
    return weights


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer=None, scheduler=None):
    """
    Load model checkpoint with error handling
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        
    Returns:
        Dictionary with epoch and other metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0),
    }
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {metadata['epoch']}")
    print(f"  Best metric: {metadata['best_metric']:.4f}")
    
    return metadata


"""
Dataset Module - Medical Multimodal Chain-of-Thought Dataset

Supports loading and processing:
- Medical images (CT, MRI, Ultrasound, etc.)
- Clinical text (medical records, lab results)
- Chain-of-thought reasoning steps
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class MedicalChainOfThoughtDataset(Dataset):
    """
    Medical Multimodal Chain-of-Thought Dataset
    
    Loads medical imaging data along with clinical text and reasoning chains.
    Each sample contains:
    - Image: Medical scan (CT/MRI/X-ray/etc.)
    - Text: Patient history, physical exam, lab results
    - Chain-of-Thought: Step-by-step reasoning process
    - Diagnosis: Final diagnosis with confidence
    """
    
    def __init__(
        self,
        data_file: str,
        data_root: str = "data",
        image_size: int = 512,
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 512,
        max_cot_steps: int = 10,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_file: Path to JSON data file
            data_root: Root directory for data
            image_size: Target image size (will be resized to square)
            tokenizer_name: HuggingFace tokenizer name
            max_text_length: Maximum token length for text
            max_cot_steps: Maximum number of reasoning steps
            transform: Custom image transforms (optional)
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.max_cot_steps = max_cot_steps
        
        # Load data from JSON
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.samples = data.get('samples', [])
        
        print(f"Loaded {len(self.samples)} samples from {data_file}")
        
        # Initialize text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Setup image transforms
        if transform is None:
            self.transform = self._get_default_transform(augment)
        else:
            self.transform = transform
    
    def _get_default_transform(self, augment: bool) -> transforms.Compose:
        """Get default image transformation pipeline"""
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        full_path = self.data_root / image_path
        image = Image.open(full_path).convert('RGB')
        return self.transform(image)
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text using tokenizer"""
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def _process_chain_of_thought(self, cot_data: Dict) -> Dict[str, Any]:
        """
        Process chain-of-thought reasoning steps
        
        Extracts and encodes:
        - Step-by-step actions and observations
        - Regions of interest (bounding boxes)
        - Intermediate conclusions
        """
        reasoning_steps = cot_data.get('reasoning_steps', [])
        
        # Extract information from each step
        step_texts = []
        step_regions = []
        step_observations = []
        
        for step in reasoning_steps[:self.max_cot_steps]:
            # Construct step text
            action = step.get('action', '')
            observation = step.get('observation', '')
            reasoning = step.get('reasoning', '')
            step_text = f"{action}. {observation}"
            if reasoning:
                step_text += f" Reasoning: {reasoning}"
            step_texts.append(step_text)
            step_observations.append(observation)
            
            # Extract region of interest (bounding box)
            roi = step.get('region_of_interest', {})
            bbox = roi.get('bbox', [0, 0, 0, 0])
            step_regions.append(bbox)
        
        # Pad to fixed length
        while len(step_texts) < self.max_cot_steps:
            step_texts.append("")
            step_regions.append([0, 0, 0, 0])
            step_observations.append("")
        
        # Encode step texts
        encoded_steps = []
        for text in step_texts:
            if text:
                encoded = self._encode_text(text)
            else:
                # Empty steps filled with zeros
                encoded = {
                    'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
                }
            encoded_steps.append(encoded)
        
        # Encode intermediate conclusions
        intermediate_conclusions = cot_data.get('intermediate_conclusions', [])
        conclusions_text = " ".join(intermediate_conclusions)
        encoded_conclusions = self._encode_text(conclusions_text)
        
        return {
            'num_steps': min(len(reasoning_steps), self.max_cot_steps),
            'step_input_ids': torch.stack([s['input_ids'] for s in encoded_steps]),
            'step_attention_mask': torch.stack([s['attention_mask'] for s in encoded_steps]),
            'step_regions': torch.tensor(step_regions, dtype=torch.float32),
            'conclusions_input_ids': encoded_conclusions['input_ids'],
            'conclusions_attention_mask': encoded_conclusions['attention_mask'],
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image']['path'])
        
        # Process medical record text
        medical_record = sample['medical_record']
        full_text = f"{medical_record['history']} {medical_record['physical_exam']} {medical_record['lab_results']}"
        encoded_text = self._encode_text(full_text)
        
        # Process chain-of-thought
        cot_data = self._process_chain_of_thought(sample['chain_of_thought'])
        
        # Process final diagnosis
        diagnosis = sample['final_diagnosis']
        diagnosis_text = diagnosis['primary']
        if diagnosis.get('secondary'):
            diagnosis_text += ", " + ", ".join(diagnosis['secondary'])
        encoded_diagnosis = self._encode_text(diagnosis_text)
        
        return {
            'sample_id': sample['sample_id'],
            'image': image,
            'text_input_ids': encoded_text['input_ids'],
            'text_attention_mask': encoded_text['attention_mask'],
            'cot_num_steps': cot_data['num_steps'],
            'cot_step_input_ids': cot_data['step_input_ids'],
            'cot_step_attention_mask': cot_data['step_attention_mask'],
            'cot_step_regions': cot_data['step_regions'],
            'cot_conclusions_input_ids': cot_data['conclusions_input_ids'],
            'cot_conclusions_attention_mask': cot_data['conclusions_attention_mask'],
            'diagnosis_input_ids': encoded_diagnosis['input_ids'],
            'diagnosis_attention_mask': encoded_diagnosis['attention_mask'],
            'diagnosis_text': diagnosis_text,
            'confidence': diagnosis.get('confidence', 1.0),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader
    
    Batches multiple samples together while handling variable-length sequences.
    """
    return {
        'sample_ids': [item['sample_id'] for item in batch],
        'images': torch.stack([item['image'] for item in batch]),
        'text_input_ids': torch.stack([item['text_input_ids'] for item in batch]),
        'text_attention_mask': torch.stack([item['text_attention_mask'] for item in batch]),
        'cot_num_steps': torch.tensor([item['cot_num_steps'] for item in batch]),
        'cot_step_input_ids': torch.stack([item['cot_step_input_ids'] for item in batch]),
        'cot_step_attention_mask': torch.stack([item['cot_step_attention_mask'] for item in batch]),
        'cot_step_regions': torch.stack([item['cot_step_regions'] for item in batch]),
        'cot_conclusions_input_ids': torch.stack([item['cot_conclusions_input_ids'] for item in batch]),
        'cot_conclusions_attention_mask': torch.stack([item['cot_conclusions_attention_mask'] for item in batch]),
        'diagnosis_input_ids': torch.stack([item['diagnosis_input_ids'] for item in batch]),
        'diagnosis_attention_mask': torch.stack([item['diagnosis_attention_mask'] for item in batch]),
        'diagnosis_texts': [item['diagnosis_text'] for item in batch],
        'confidences': torch.tensor([item['confidence'] for item in batch]),
    }

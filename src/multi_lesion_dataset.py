"""
Enhanced Dataset for Multi-Lesion Support

Handles:
1. Multiple images per sample (e.g., T1/T2 MRI, multi-phase CT)
2. Multiple lesions per image with segmentation masks
3. Per-lesion annotations and reasoning
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


class MultiLesionMedicalDataset(Dataset):
    """
    Enhanced dataset supporting multiple lesions and multiple images
    """
    
    def __init__(
        self,
        data_file: str,
        data_root: str = "data",
        image_size: int = 512,
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 512,
        max_cot_steps: int = 10,
        max_num_images: int = 3,
        max_lesions_per_sample: int = 10,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        load_segmentation_masks: bool = True,
    ):
        """
        Args:
            data_file: JSON data file path
            data_root: Data root directory
            image_size: Target image size
            tokenizer_name: Text tokenizer name
            max_text_length: Maximum text length
            max_cot_steps: Maximum reasoning steps
            max_num_images: Maximum number of images per sample
            max_lesions_per_sample: Maximum lesions per sample
            transform: Image transforms
            augment: Data augmentation flag
            load_segmentation_masks: Whether to load segmentation masks
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.max_cot_steps = max_cot_steps
        self.max_num_images = max_num_images
        self.max_lesions_per_sample = max_lesions_per_sample
        self.load_segmentation_masks = load_segmentation_masks
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.samples = data.get('samples', [])
        
        print(f"Loaded {len(self.samples)} samples from {data_file}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Setup transforms
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
        """Load and transform single image"""
        full_path = self.data_root / image_path
        image = Image.open(full_path).convert('RGB')
        return self.transform(image)
    
    def _load_images(self, image_info: Dict) -> torch.Tensor:
        """
        Load single or multiple images
        
        Args:
            image_info: Can be:
                - Single image: {'path': 'img.jpg', ...}
                - Multiple images: {'paths': ['img1.jpg', 'img2.jpg'], ...}
        
        Returns:
            images: [3, H, W] or [num_images, 3, H, W]
        """
        if 'paths' in image_info:
            # Multiple images
            image_paths = image_info['paths'][:self.max_num_images]
            images = []
            for path in image_paths:
                img = self._load_image(path)
                images.append(img)
            
            # Pad to max_num_images
            while len(images) < self.max_num_images:
                images.append(torch.zeros_like(images[0]))
            
            return torch.stack(images, dim=0)  # [num_images, 3, H, W]
        else:
            # Single image
            return self._load_image(image_info['path'])  # [3, H, W]
    
    def _load_segmentation_mask(self, mask_path: str) -> torch.Tensor:
        """Load segmentation mask"""
        full_path = self.data_root / mask_path
        
        if mask_path.endswith('.npy'):
            mask = np.load(full_path)
        else:
            mask = np.array(Image.open(full_path))
        
        # Resize to target size
        mask = cv2.resize(mask, (self.image_size, self.image_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        return torch.from_numpy(mask).long()
    
    def _process_lesions(self, sample: Dict, sample_idx: int) -> Dict:
        """
        Process lesion information
        
        Returns:
            Dictionary with:
                - lesion_bboxes: [num_lesions, 5] (batch_idx, x1, y1, x2, y2)
                - lesion_masks: [num_lesions, H, W]
                - lesion_labels: [num_lesions]
                - lesion_confidences: [num_lesions]
        """
        lesions = sample.get('lesions', [])
        
        lesion_bboxes = []
        lesion_masks = []
        lesion_labels = []
        lesion_confidences = []
        lesion_descriptions = []
        
        for lesion in lesions[:self.max_lesions_per_sample]:
            # Bounding box (add batch index as first element)
            bbox = lesion.get('bbox', [0, 0, 0, 0])
            lesion_bboxes.append([0] + bbox)  # Batch index will be set in collate_fn
            
            # Segmentation mask (if available and needed)
            if self.load_segmentation_masks and 'segmentation_mask' in lesion:
                mask = self._load_segmentation_mask(lesion['segmentation_mask'])
                lesion_masks.append(mask)
            else:
                # Create mask from bbox
                mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
                x1, y1, x2, y2 = bbox
                # Scale bbox to image_size
                scale = self.image_size / 512  # Assuming original coords are for 512x512
                x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                mask[y1:y2, x1:x2] = lesion.get('lesion_id', 1)
                lesion_masks.append(mask)
            
            # Label and confidence
            lesion_type = lesion.get('type', 'unknown')
            lesion_labels.append(lesion_type)
            
            confidence = lesion.get('confidence', 1.0)
            lesion_confidences.append(confidence)
            
            description = lesion.get('description', '')
            lesion_descriptions.append(description)
        
        # Padding
        num_lesions = len(lesion_bboxes)
        while len(lesion_bboxes) < self.max_lesions_per_sample:
            lesion_bboxes.append([0, 0, 0, 0, 0])
            lesion_masks.append(torch.zeros(self.image_size, self.image_size, dtype=torch.long))
            lesion_labels.append('none')
            lesion_confidences.append(0.0)
            lesion_descriptions.append('')
        
        return {
            'num_lesions': num_lesions,
            'lesion_bboxes': torch.tensor(lesion_bboxes, dtype=torch.float32),
            'lesion_masks': torch.stack(lesion_masks) if lesion_masks else None,
            'lesion_labels': lesion_labels,
            'lesion_confidences': torch.tensor(lesion_confidences, dtype=torch.float32),
            'lesion_descriptions': lesion_descriptions,
        }
    
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
        """Process chain-of-thought with lesion associations"""
        reasoning_steps = cot_data.get('reasoning_steps', [])
        
        step_texts = []
        step_regions = []
        step_lesion_ids = []
        
        for step in reasoning_steps[:self.max_cot_steps]:
            # Construct step text
            action = step.get('action', '')
            observation = step.get('observation', '')
            reasoning = step.get('reasoning', '')
            step_text = f"{action}. {observation}"
            if reasoning:
                step_text += f" Reasoning: {reasoning}"
            step_texts.append(step_text)
            
            # Extract region of interest
            roi = step.get('region_of_interest', {})
            bbox = roi.get('bbox', [0, 0, 0, 0])
            step_regions.append(bbox)
            
            # Lesion association
            lesion_id = step.get('lesion_id', -1)  # -1 means no specific lesion
            step_lesion_ids.append(lesion_id)
        
        # Padding
        while len(step_texts) < self.max_cot_steps:
            step_texts.append("")
            step_regions.append([0, 0, 0, 0])
            step_lesion_ids.append(-1)
        
        # Encode step texts
        encoded_steps = []
        for text in step_texts:
            if text:
                encoded = self._encode_text(text)
            else:
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
            'step_lesion_ids': torch.tensor(step_lesion_ids, dtype=torch.long),
            'conclusions_input_ids': encoded_conclusions['input_ids'],
            'conclusions_attention_mask': encoded_conclusions['attention_mask'],
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with multi-lesion support"""
        sample = self.samples[idx]
        
        # Load images (single or multiple)
        images = self._load_images(sample['image'])
        
        # Process medical record text
        medical_record = sample['medical_record']
        full_text = f"{medical_record['history']} {medical_record['physical_exam']} {medical_record['lab_results']}"
        encoded_text = self._encode_text(full_text)
        
        # Process lesions
        lesion_data = self._process_lesions(sample, idx)
        
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
            'images': images,
            'text_input_ids': encoded_text['input_ids'],
            'text_attention_mask': encoded_text['attention_mask'],
            'cot_num_steps': cot_data['num_steps'],
            'cot_step_input_ids': cot_data['step_input_ids'],
            'cot_step_attention_mask': cot_data['step_attention_mask'],
            'cot_step_regions': cot_data['step_regions'],
            'cot_step_lesion_ids': cot_data['step_lesion_ids'],
            'num_lesions': lesion_data['num_lesions'],
            'lesion_bboxes': lesion_data['lesion_bboxes'],
            'lesion_masks': lesion_data['lesion_masks'],
            'lesion_labels': lesion_data['lesion_labels'],
            'lesion_confidences': lesion_data['lesion_confidences'],
            'diagnosis_text': diagnosis_text,
            'global_confidence': diagnosis.get('confidence', 1.0),
        }


def multi_lesion_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for multi-lesion batches
    
    Handles variable number of lesions across samples
    """
    batch_size = len(batch)
    
    # Stack images (handle both single and multi-image cases)
    images = batch[0]['images']
    if len(images.shape) == 3:  # Single image [3, H, W]
        stacked_images = torch.stack([item['images'] for item in batch])
    else:  # Multiple images [num_images, 3, H, W]
        stacked_images = torch.stack([item['images'] for item in batch])
    
    # Collect all lesions with batch indices
    all_lesion_bboxes = []
    all_lesion_labels = []
    all_lesion_confidences = []
    lesion_to_sample = []  # Track which sample each lesion belongs to
    lesion_to_step = []  # Track which CoT step each lesion is associated with
    
    for batch_idx, item in enumerate(batch):
        num_lesions = item['num_lesions']
        
        for lesion_idx in range(num_lesions):
            # Set batch index
            bbox = item['lesion_bboxes'][lesion_idx].clone()
            bbox[0] = batch_idx
            all_lesion_bboxes.append(bbox)
            
            all_lesion_labels.append(item['lesion_labels'][lesion_idx])
            all_lesion_confidences.append(item['lesion_confidences'][lesion_idx])
            lesion_to_sample.append(batch_idx)
            
            # Find which CoT step this lesion is associated with
            step_lesion_ids = item['cot_step_lesion_ids']
            # Find first step that mentions this lesion
            associated_step = -1
            for step_idx, step_lesion_id in enumerate(step_lesion_ids):
                if step_lesion_id == lesion_idx:
                    associated_step = step_idx
                    break
            lesion_to_step.append(associated_step)
    
    # Stack lesion data
    if all_lesion_bboxes:
        lesion_bboxes = torch.stack(all_lesion_bboxes)
        lesion_confidences = torch.tensor(all_lesion_confidences)
        lesion_to_sample = torch.tensor(lesion_to_sample)
        lesion_to_step = torch.tensor(lesion_to_step)
    else:
        lesion_bboxes = None
        all_lesion_labels = []
        lesion_confidences = None
        lesion_to_sample = None
        lesion_to_step = None
    
    return {
        'sample_ids': [item['sample_id'] for item in batch],
        'images': stacked_images,
        'text_input_ids': torch.stack([item['text_input_ids'] for item in batch]),
        'text_attention_mask': torch.stack([item['text_attention_mask'] for item in batch]),
        'cot_num_steps': torch.tensor([item['cot_num_steps'] for item in batch]),
        'cot_step_input_ids': torch.stack([item['cot_step_input_ids'] for item in batch]),
        'cot_step_attention_mask': torch.stack([item['cot_step_attention_mask'] for item in batch]),
        'cot_step_regions': torch.stack([item['cot_step_regions'] for item in batch]),
        'cot_step_lesion_ids': torch.stack([item['cot_step_lesion_ids'] for item in batch]),
        'num_lesions': torch.tensor([item['num_lesions'] for item in batch]),
        'lesion_bboxes': lesion_bboxes,
        'lesion_labels': all_lesion_labels,
        'lesion_confidences': lesion_confidences,
        'lesion_to_sample': lesion_to_sample,
        'lesion_to_step': lesion_to_step,
        'diagnosis_texts': [item['diagnosis_text'] for item in batch],
        'global_confidences': torch.tensor([item['global_confidence'] for item in batch]),
    }


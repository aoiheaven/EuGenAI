"""
Training Script for Medical Multimodal Chain-of-Thought Model

Supports:
- Multi-GPU training
- Mixed precision training
- TensorBoard/WandB logging
- Automatic checkpointing
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataset import MedicalChainOfThoughtDataset, collate_fn
from model import MedicalMultimodalCoT
from utils import DiagnosisLabelEncoder, validate_config, ensure_directories


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """Training manager for medical multimodal model"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Validate configuration
        validate_config(config)
        
        # Create directories
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diagnosis label encoder
        label_file = config.get('dataset', {}).get('label_file', 'data/diagnosis_labels.json')
        self.label_encoder = DiagnosisLabelEncoder()
        
        # Build vocabulary from training data if label file doesn't exist
        if not os.path.exists(label_file):
            print(f"Building diagnosis vocabulary from training data...")
            self.label_encoder.build_from_dataset(
                config['dataset']['train_file'],
                save_path=label_file
            )
        else:
            print(f"Loading diagnosis vocabulary from {label_file}")
            self.label_encoder.load_from_file(label_file)
        
        # Update model config with correct number of classes
        model_config = config['model'].copy()
        model_config['num_diagnosis_classes'] = max(self.label_encoder.num_classes, 2)  # At least 2 classes
        
        # Initialize model
        self.model = MedicalMultimodalCoT(**model_config).to(self.device)
        print(f"Model initialized with {self.label_encoder.num_classes} diagnosis classes")
        
        # Initialize datasets
        self.train_dataset = MedicalChainOfThoughtDataset(
            data_file=config['dataset']['train_file'],
            data_root=config['dataset']['data_root'],
            image_size=config['dataset']['image_size'],
            max_text_length=config['dataset']['max_text_length'],
            max_cot_steps=config['dataset']['max_cot_steps'],
            augment=config['dataset']['augment_train'],
        )
        
        self.val_dataset = MedicalChainOfThoughtDataset(
            data_file=config['dataset']['val_file'],
            data_root=config['dataset']['data_root'],
            image_size=config['dataset']['image_size'],
            max_text_length=config['dataset']['max_text_length'],
            max_cot_steps=config['dataset']['max_cot_steps'],
            augment=config['dataset']['augment_val'],
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'],
            collate_fn=collate_fn,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'],
            collate_fn=collate_fn,
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['training']['betas'],
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr'],
        )
        
        # Initialize loss functions
        self.criterion_diagnosis = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding
        self.criterion_confidence = nn.MSELoss()
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config['hardware']['mixed_precision'] else None
        
        # Logging
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_diagnosis_loss = 0.0
        epoch_confidence_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cot_step_input_ids = batch['cot_step_input_ids'].to(self.device)
            cot_step_attention_mask = batch['cot_step_attention_mask'].to(self.device)
            cot_step_regions = batch['cot_step_regions'].to(self.device)
            cot_num_steps = batch['cot_num_steps'].to(self.device)
            confidences = batch['confidences'].to(self.device)
            
            # Encode diagnosis labels
            diagnosis_labels = self.label_encoder.encode_batch(batch['diagnosis_texts']).to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images=images,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        cot_step_input_ids=cot_step_input_ids,
                        cot_step_attention_mask=cot_step_attention_mask,
                        cot_step_regions=cot_step_regions,
                        cot_num_steps=cot_num_steps,
                    )
                    
                    # Compute losses
                    diagnosis_loss = self.criterion_diagnosis(
                        outputs['diagnosis_logits'], diagnosis_labels
                    )
                    confidence_loss = self.criterion_confidence(
                        outputs['confidence'].squeeze(), confidences
                    )
                    
                    total_loss = (
                        self.loss_weights['diagnosis'] * diagnosis_loss +
                        self.loss_weights['confidence'] * confidence_loss
                    )
                
                # Backward pass with scaling
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    images=images,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    cot_step_input_ids=cot_step_input_ids,
                    cot_step_attention_mask=cot_step_attention_mask,
                    cot_step_regions=cot_step_regions,
                    cot_num_steps=cot_num_steps,
                )
                
                diagnosis_loss = self.criterion_diagnosis(
                    outputs['diagnosis_logits'], diagnosis_labels
                )
                confidence_loss = self.criterion_confidence(
                    outputs['confidence'].squeeze(), confidences
                )
                
                total_loss = (
                    self.loss_weights['diagnosis'] * diagnosis_loss +
                    self.loss_weights['confidence'] * confidence_loss
                )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_diagnosis_loss += diagnosis_loss.item()
            epoch_confidence_loss += confidence_loss.item()
            
            # Logging
            if batch_idx % self.config['logging']['log_interval'] == 0:
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'conf_loss': confidence_loss.item(),
                })
                
                if self.writer:
                    self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                    self.writer.add_scalar('train/conf_loss', confidence_loss.item(), self.global_step)
            
            self.global_step += 1
        
        return {
            'loss': epoch_loss / len(self.train_loader),
            'diagnosis_loss': epoch_diagnosis_loss / len(self.train_loader),
            'confidence_loss': epoch_confidence_loss / len(self.train_loader),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_loss = 0.0
        val_confidence_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cot_step_input_ids = batch['cot_step_input_ids'].to(self.device)
            cot_step_attention_mask = batch['cot_step_attention_mask'].to(self.device)
            cot_step_regions = batch['cot_step_regions'].to(self.device)
            cot_num_steps = batch['cot_num_steps'].to(self.device)
            confidences = batch['confidences'].to(self.device)
            
            # Encode diagnosis labels
            diagnosis_labels = self.label_encoder.encode_batch(batch['diagnosis_texts']).to(self.device)
            
            outputs = self.model(
                images=images,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                cot_step_input_ids=cot_step_input_ids,
                cot_step_attention_mask=cot_step_attention_mask,
                cot_step_regions=cot_step_regions,
                cot_num_steps=cot_num_steps,
            )
            
            diagnosis_loss = self.criterion_diagnosis(
                outputs['diagnosis_logits'], diagnosis_labels
            )
            confidence_loss = self.criterion_confidence(
                outputs['confidence'].squeeze(), confidences
            )
            
            total_loss = (
                self.loss_weights['diagnosis'] * diagnosis_loss +
                self.loss_weights['confidence'] * confidence_loss
            )
            
            val_loss += total_loss.item()
            val_confidence_loss += confidence_loss.item()
        
        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_confidence_loss': val_confidence_loss / len(self.val_loader),
        }
    
    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Checkpoint saved: {filename}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if (epoch + 1) % self.config['validation']['val_interval'] == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch + 1} - Val Loss: {val_metrics['val_loss']:.4f}")
                
                if self.writer:
                    self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
                
                # Save best model
                if val_metrics['val_loss'] < self.best_metric or self.best_metric == 0:
                    self.best_metric = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint()
            
            # Update scheduler
            self.scheduler.step()
        
        print("Training completed!")
        if self.writer:
            self.writer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Medical Multimodal CoT Model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()


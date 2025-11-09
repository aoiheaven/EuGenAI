"""
Self-Supervised Learning Module for EuGenAI

This module implements self-supervised pre-training strategies including:
- Contrastive learning (CLIP-style image-text alignment)
- Masked image modeling (MAE-style)
- Masked language modeling (BERT-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning module for image-text alignment.
    Implements InfoNCE loss similar to CLIP.
    """
    
    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection heads
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            image_features: [batch_size, image_dim]
            text_features: [batch_size, text_dim]
            
        Returns:
            loss: Contrastive loss
            metrics: Dictionary of metrics
        """
        batch_size = image_features.shape[0]
        
        # Project to common space
        image_embeds = self.image_projection(image_features)
        text_embeds = self.text_projection(text_features)
        
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)
        
        # InfoNCE loss (symmetric)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        # Compute accuracy
        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.t().argmax(dim=1) == labels).float().mean()
            acc = (i2t_acc + t2i_acc) / 2
        
        metrics = {
            'contrastive_loss': loss.item(),
            'i2t_accuracy': i2t_acc.item(),
            't2i_accuracy': t2i_acc.item(),
            'avg_accuracy': acc.item(),
            'logit_scale': 1 / self.temperature
        }
        
        return loss, metrics


class MaskedImageModeling(nn.Module):
    """
    Masked image modeling for self-supervised learning.
    Predicts masked patches similar to MAE.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        patch_size: int = 16,
        num_channels: int = 3,
        decoder_dim: int = 512,
        decoder_depth: int = 4
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        # Decoder for reconstruction
        self.decoder_embed = nn.Linear(hidden_dim, decoder_dim)
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=8,
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(decoder_depth)
        ])
        
        # Predict pixel values
        self.decoder_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * num_channels
        )
        
    def forward(
        self,
        encoded_patches: torch.Tensor,
        original_patches: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            encoded_patches: [batch_size, num_patches, hidden_dim]
            original_patches: [batch_size, num_patches, patch_size^2 * C]
            mask: [batch_size, num_patches] - 1 for masked, 0 for visible
            
        Returns:
            loss: Reconstruction loss on masked patches
            metrics: Dictionary of metrics
        """
        batch_size, num_patches, _ = encoded_patches.shape
        
        # Decode
        x = self.decoder_embed(encoded_patches)
        
        for block in self.decoder_blocks:
            x = block(x)
        
        # Predict pixel values
        pred_patches = self.decoder_pred(x)  # [B, N, patch_size^2 * C]
        
        # Compute loss only on masked patches
        loss = F.mse_loss(
            pred_patches[mask.bool()],
            original_patches[mask.bool()],
            reduction='mean'
        )
        
        # Metrics
        with torch.no_grad():
            # Compute PSNR on masked patches
            mse = F.mse_loss(
                pred_patches[mask.bool()],
                original_patches[mask.bool()],
                reduction='mean'
            )
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            
            # Mask ratio
            mask_ratio = mask.float().mean()
        
        metrics = {
            'reconstruction_loss': loss.item(),
            'psnr': psnr.item(),
            'mask_ratio': mask_ratio.item()
        }
        
        return loss, metrics


class MaskedLanguageModeling(nn.Module):
    """
    Masked language modeling for clinical text.
    Similar to BERT's MLM objective.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 30522  # BERT vocab size
    ):
        super().__init__()
        
        # Prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(
        self,
        encoded_tokens: torch.Tensor,
        original_token_ids: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            encoded_tokens: [batch_size, seq_len, hidden_dim]
            original_token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len] - 1 for masked, 0 for visible
            
        Returns:
            loss: MLM loss on masked tokens
            metrics: Dictionary of metrics
        """
        # Predict token IDs
        logits = self.mlm_head(encoded_tokens)  # [B, L, vocab_size]
        
        # Compute loss only on masked tokens
        loss = F.cross_entropy(
            logits[mask.bool()],
            original_token_ids[mask.bool()],
            reduction='mean'
        )
        
        # Accuracy on masked tokens
        with torch.no_grad():
            pred_tokens = logits.argmax(dim=-1)
            acc = (pred_tokens[mask.bool()] == original_token_ids[mask.bool()]).float().mean()
            mask_ratio = mask.float().mean()
        
        metrics = {
            'mlm_loss': loss.item(),
            'mlm_accuracy': acc.item(),
            'text_mask_ratio': mask_ratio.item()
        }
        
        return loss, metrics


class SelfSupervisedLearner:
    """
    Main self-supervised learning coordinator.
    Combines multiple SSL tasks for pre-training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize SSL modules
        if config['self_supervised']['contrastive']['enabled']:
            self.contrastive = ContrastiveLearning(
                image_dim=config['model']['hidden_dim'],
                text_dim=config['model']['hidden_dim'],
                projection_dim=config['model'].get('contrastive_projection_dim', 256),
                temperature=config['model'].get('contrastive_temperature', 0.07)
            ).to(device)
        
        if config['self_supervised']['masked_image']['enabled']:
            self.masked_image = MaskedImageModeling(
                hidden_dim=config['model']['hidden_dim'],
                patch_size=config['model']['img_size'] // 32,  # Assuming 32 patches
                decoder_depth=4
            ).to(device)
        
        if config['self_supervised']['masked_text']['enabled']:
            self.masked_text = MaskedLanguageModeling(
                hidden_dim=config['model']['hidden_dim']
            ).to(device)
    
    def compute_loss(
        self,
        batch: Dict,
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute combined self-supervised loss.
        
        Args:
            batch: Dictionary containing inputs
            return_metrics: Whether to return detailed metrics
            
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of metrics (if return_metrics=True)
        """
        total_loss = 0
        metrics = {}
        
        # Extract features from model
        image_features = self.model.image_encoder(batch['image'])
        text_features = self.model.text_encoder(
            batch['text_input_ids'],
            batch['text_attention_mask']
        )
        
        # Contrastive learning
        if hasattr(self, 'contrastive'):
            cont_loss, cont_metrics = self.contrastive(
                image_features.mean(dim=1),  # Global pooling
                text_features.mean(dim=1)
            )
            weight = self.config['training']['loss_weights']['contrastive']
            total_loss += weight * cont_loss
            metrics.update(cont_metrics)
        
        # Masked image modeling
        if hasattr(self, 'masked_image') and 'masked_image_patches' in batch:
            mim_loss, mim_metrics = self.masked_image(
                image_features,
                batch['original_patches'],
                batch['image_mask']
            )
            weight = self.config['training']['loss_weights']['masked_region']
            total_loss += weight * mim_loss
            metrics.update(mim_metrics)
        
        # Masked language modeling
        if hasattr(self, 'masked_text') and 'masked_text' in batch:
            mlm_loss, mlm_metrics = self.masked_text(
                text_features,
                batch['original_token_ids'],
                batch['text_mask']
            )
            weight = self.config['training']['loss_weights']['masked_text']
            total_loss += weight * mlm_loss
            metrics.update(mlm_metrics)
        
        metrics['total_loss'] = total_loss.item()
        
        if return_metrics:
            return total_loss, metrics
        return total_loss, None


def create_mask(
    tensor: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_strategy: str = 'random'
) -> torch.Tensor:
    """
    Create mask for self-supervised learning.
    
    Args:
        tensor: Input tensor [batch_size, seq_len, ...]
        mask_ratio: Percentage to mask
        mask_strategy: 'random', 'block', or 'semantic'
        
    Returns:
        mask: Binary mask [batch_size, seq_len]
    """
    batch_size, seq_len = tensor.shape[:2]
    
    if mask_strategy == 'random':
        # Random masking
        mask = torch.rand(batch_size, seq_len, device=tensor.device) < mask_ratio
    
    elif mask_strategy == 'block':
        # Block masking
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tensor.device)
        num_blocks = int(seq_len * mask_ratio / 4)  # Assuming block size of 4
        for i in range(batch_size):
            indices = torch.randperm(seq_len - 3)[:num_blocks]
            for idx in indices:
                mask[i, idx:idx+4] = True
    
    else:  # semantic
        # Semantic masking (mask entire semantic regions)
        # This requires additional semantic segmentation info
        mask = torch.rand(batch_size, seq_len, device=tensor.device) < mask_ratio
    
    return mask


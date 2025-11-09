"""
Model Architecture - Multimodal Transformer for Chain-of-Thought Reasoning

Combines:
- Vision Transformer for medical image encoding
- BERT-based text encoder for clinical text
- Cross-modal attention for reasoning
- Chain-of-thought decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import BertModel, BertConfig
import timm


class ImageEncoder(nn.Module):
    """
    Vision Transformer for medical image encoding
    
    Uses a pre-trained ViT backbone and extracts hierarchical features
    for both global understanding and region-specific analysis.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 0,
        img_size: int = 512,
    ):
        """
        Args:
            model_name: timm model name
            pretrained: whether to load pretrained weights
            num_classes: number of output classes (0 for feature extraction)
            img_size: input image size
        """
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.embed_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Image tensor [B, 3, H, W]
            
        Returns:
            global_features: [B, feature_dim]
            patch_features: [B, num_patches, feature_dim]
        """
        # Extract features from backbone
        features = self.backbone.forward_features(x)
        
        if hasattr(self.backbone, 'fc_norm'):
            # For models with fc_norm (like ViT)
            global_features = self.backbone.fc_norm(features.mean(1))
            patch_features = features
        else:
            global_features = features.mean(1)
            patch_features = features
            
        return global_features, patch_features


class TextEncoder(nn.Module):
    """
    BERT-based encoder for clinical text
    
    Encodes medical records, observations, and reasoning steps.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model name
            freeze_embeddings: whether to freeze embedding layers
        """
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.feature_dim = self.bert.config.hidden_size
        
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            pooled_output: [B, feature_dim]
            sequence_output: [B, seq_len, feature_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return outputs.pooler_output, outputs.last_hidden_state


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for image-text fusion
    
    Allows image patches to attend to text tokens and vice versa.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query features [B, N_q, dim]
            key: Key features [B, N_k, dim]
            value: Value features [B, N_v, dim]
            key_padding_mask: Mask for keys [B, N_k]
            
        Returns:
            output: Attended features [B, N_q, dim]
            attention_weights: [B, N_q, N_k]
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )
        
        # Residual + norm
        query = self.norm1(query + attn_output)
        
        # FFN
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output, attn_weights


class ChainOfThoughtDecoder(nn.Module):
    """
    Decoder for chain-of-thought reasoning
    
    Processes reasoning steps sequentially and generates predictions.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_steps: int = 10,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Feature dimension
            num_steps: Maximum number of reasoning steps
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_steps = num_steps
        
        # Step positional encoding
        self.step_embedding = nn.Embedding(num_steps, dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        step_features: torch.Tensor,
        memory: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            step_features: Encoded reasoning steps [B, num_steps, dim]
            memory: Context from image and text [B, seq_len, dim]
            step_mask: Mask for steps [B, num_steps]
            memory_mask: Mask for memory [B, seq_len]
            
        Returns:
            output: Decoded features [B, num_steps, dim]
        """
        B, N, D = step_features.shape
        
        # Add positional encoding
        positions = torch.arange(N, device=step_features.device).unsqueeze(0).expand(B, -1)
        step_features = step_features + self.step_embedding(positions)
        
        # Transformer decoding
        output = self.transformer_decoder(
            tgt=step_features,
            memory=memory,
            tgt_key_padding_mask=step_mask,
            memory_key_padding_mask=memory_mask,
        )
        
        return self.norm(output)


class MedicalMultimodalCoT(nn.Module):
    """
    Main model: Medical Multimodal Chain-of-Thought
    
    End-to-end model that:
    1. Encodes medical images and clinical text
    2. Performs cross-modal reasoning
    3. Generates chain-of-thought explanations
    4. Predicts diagnosis with confidence
    """
    
    def __init__(
        self,
        image_encoder_name: str = "vit_base_patch16_224",
        text_encoder_name: str = "bert-base-uncased",
        img_size: int = 512,
        hidden_dim: int = 768,
        num_cot_steps: int = 10,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        num_diagnosis_classes: int = 100,
    ):
        """
        Args:
            image_encoder_name: Vision model name
            text_encoder_name: Text model name
            img_size: Input image size
            hidden_dim: Hidden dimension
            num_cot_steps: Number of reasoning steps
            num_heads: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            num_diagnosis_classes: Number of diagnosis classes
        """
        super().__init__()
        
        # Encoders
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            img_size=img_size,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
        )
        
        # Project to common dimension
        self.image_proj = nn.Linear(self.image_encoder.feature_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.feature_dim, hidden_dim)
        
        # Cross-modal attention
        self.img_to_text_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.text_to_img_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Chain-of-thought decoder
        self.cot_decoder = ChainOfThoughtDecoder(
            dim=hidden_dim,
            num_steps=num_cot_steps,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )
        
        # Region attention (for localizing reasoning to image regions)
        self.region_attention = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for bbox coordinates
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Prediction heads
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diagnosis_classes),
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        cot_step_input_ids: torch.Tensor,
        cot_step_attention_mask: torch.Tensor,
        cot_step_regions: torch.Tensor,
        cot_num_steps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Medical images [B, 3, H, W]
            text_input_ids: Clinical text tokens [B, seq_len]
            text_attention_mask: Text attention mask [B, seq_len]
            cot_step_input_ids: Reasoning step tokens [B, num_steps, seq_len]
            cot_step_attention_mask: Step attention masks [B, num_steps, seq_len]
            cot_step_regions: Bounding boxes [B, num_steps, 4]
            cot_num_steps: Actual number of steps per sample [B]
            
        Returns:
            Dictionary containing:
                - diagnosis_logits: [B, num_classes]
                - confidence: [B, 1]
                - step_attentions: [B, num_steps, num_patches]
                - cross_modal_attention: [B, num_patches, seq_len]
        """
        B = images.size(0)
        
        # Encode image
        img_global, img_patches = self.image_encoder(images)
        img_patches = self.image_proj(img_patches)
        
        # Encode clinical text
        text_pooled, text_sequence = self.text_encoder(
            text_input_ids,
            text_attention_mask,
        )
        text_sequence = self.text_proj(text_sequence)
        
        # Cross-modal attention
        img_attended, img_to_text_attn = self.img_to_text_attn(
            query=img_patches,
            key=text_sequence,
            value=text_sequence,
            key_padding_mask=~text_attention_mask.bool(),
        )
        
        text_attended, text_to_img_attn = self.text_to_img_attn(
            query=text_sequence,
            key=img_patches,
            value=img_patches,
        )
        
        # Combine context
        context = torch.cat([img_attended, text_attended], dim=1)
        context_mask = torch.cat([
            torch.ones(B, img_attended.size(1), device=images.device, dtype=torch.bool),
            text_attention_mask.bool(),
        ], dim=1)
        
        # Encode reasoning steps
        num_steps = cot_step_input_ids.size(1)
        step_features = []
        
        for i in range(num_steps):
            step_pooled, _ = self.text_encoder(
                cot_step_input_ids[:, i, :],
                cot_step_attention_mask[:, i, :],
            )
            step_features.append(self.text_proj(step_pooled))
        
        step_features = torch.stack(step_features, dim=1)  # [B, num_steps, dim]
        
        # Create step mask
        step_mask = torch.arange(num_steps, device=images.device).unsqueeze(0) >= cot_num_steps.unsqueeze(1)
        
        # Decode chain-of-thought
        cot_output = self.cot_decoder(
            step_features=step_features,
            memory=context,
            step_mask=step_mask,
            memory_mask=~context_mask,
        )
        
        # Compute region-specific attention
        regions_normalized = cot_step_regions / images.size(-1)  # Normalize bbox coordinates
        region_features = torch.cat([cot_output, regions_normalized], dim=-1)
        region_attn_scores = self.region_attention(region_features).squeeze(-1)  # [B, num_steps]
        
        # Aggregate reasoning
        final_reasoning = (cot_output * F.softmax(region_attn_scores, dim=1).unsqueeze(-1)).sum(dim=1)
        
        # Predictions
        diagnosis_logits = self.diagnosis_classifier(final_reasoning)
        confidence = self.confidence_predictor(final_reasoning)
        
        return {
            'diagnosis_logits': diagnosis_logits,
            'confidence': confidence,
            'step_attentions': region_attn_scores,
            'cross_modal_attention': img_to_text_attn,
            'reasoning_features': cot_output,
        }


def create_model(config: Dict) -> MedicalMultimodalCoT:
    """
    Factory function to create model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    return MedicalMultimodalCoT(**config)


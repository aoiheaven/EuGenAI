"""
Enhanced Multi-Lesion Model Architecture

Supports:
1. Multi-lesion segmentation and detection
2. Multiple image inputs (e.g., different MRI sequences, temporal CT scans)
3. Per-lesion attention and reasoning
4. Lesion-specific diagnosis + global diagnosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision.ops import RoIAlign

from model import ImageEncoder, TextEncoder, CrossModalAttention, ChainOfThoughtDecoder


class MultiLesionSegmentationHead(nn.Module):
    """
    Segmentation decoder for multi-lesion detection
    
    Outputs pixel-level segmentation masks for different lesion types
    and instance segmentation to separate individual lesions.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_lesion_types: int = 10,
        img_size: int = 512,
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_lesion_types: Number of lesion categories
            img_size: Output image size
        """
        super().__init__()
        
        # Calculate upsampling stages based on ViT patch size (typically 16)
        # patch_size=16 means 512/16=32 patches per side
        # Need to upsample from 32x32 to 512x512 (16x)
        
        self.upsample_layers = nn.ModuleList([
            # 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(input_dim, 512, kernel_size=2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            # 64x64 -> 128x128
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # 128x128 -> 256x256
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # 256x256 -> 512x512
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Semantic segmentation head (lesion type classification per pixel)
        self.semantic_head = nn.Conv2d(64, num_lesion_types + 1, kernel_size=1)  # +1 for background
        
        # Instance segmentation head (separate individual lesions)
        self.instance_head = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            patch_features: [B, num_patches, dim] from ViT
            
        Returns:
            seg_logits: [B, num_lesion_types+1, H, W]
            instance_map: [B, 1, H, W]
        """
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)
        
        # Reshape patches to spatial feature map
        x = patch_features.transpose(1, 2).reshape(B, D, H, W)
        
        # Progressive upsampling
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Segmentation outputs
        seg_logits = self.semantic_head(x)
        instance_map = torch.sigmoid(self.instance_head(x))
        
        return seg_logits, instance_map


class LesionROIExtractor(nn.Module):
    """
    Extract features for each lesion using RoI pooling
    
    This allows per-lesion reasoning and diagnosis.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        output_dim: int = 512,
        roi_size: int = 7,
    ):
        """
        Args:
            feature_dim: Input feature dimension
            output_dim: Output feature dimension per lesion
            roi_size: RoI pooling output size
        """
        super().__init__()
        
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
        )
        
        # Feature aggregation for each lesion
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim * roi_size * roi_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
        )
        
    def forward(
        self,
        feature_map: torch.Tensor,
        lesion_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract per-lesion features
        
        Args:
            feature_map: [B, H, W, feature_dim] spatial features
            lesion_bboxes: [N, 5] where each row is [batch_idx, x1, y1, x2, y2]
            
        Returns:
            lesion_features: [N, output_dim]
        """
        # Permute to [B, C, H, W] for RoIAlign
        feature_map = feature_map.permute(0, 3, 1, 2)
        
        # RoI pooling for each lesion
        roi_features = self.roi_align(feature_map, lesion_bboxes)  # [N, feature_dim, roi_size, roi_size]
        
        # Flatten and project
        roi_features_flat = roi_features.flatten(1)  # [N, feature_dim * roi_size^2]
        lesion_features = self.feature_mlp(roi_features_flat)  # [N, output_dim]
        
        return lesion_features


class MultiImageFusion(nn.Module):
    """
    Fuse features from multiple images (e.g., different MRI sequences)
    
    Supports temporal analysis (e.g., pre/post contrast, different time points)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_images: int = 3,
        fusion_method: str = 'attention',
    ):
        """
        Args:
            hidden_dim: Feature dimension
            num_images: Maximum number of images
            fusion_method: 'attention', 'concat', or 'average'
        """
        super().__init__()
        
        self.num_images = num_images
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            # Learnable attention weights for each image
            self.image_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif fusion_method == 'concat':
            # Concatenate and project
            self.fusion_proj = nn.Linear(hidden_dim * num_images, hidden_dim)
    
    def forward(self, image_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple image features
        
        Args:
            image_features_list: List of [B, num_patches, dim] tensors
            
        Returns:
            fused_features: [B, num_patches, dim]
        """
        if self.fusion_method == 'average':
            # Simple averaging
            stacked = torch.stack(image_features_list, dim=0)  # [num_images, B, N, D]
            fused = stacked.mean(dim=0)
            
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            stacked = torch.stack(image_features_list, dim=2)  # [B, N, num_images, D]
            B, N, M, D = stacked.shape
            
            # Compute attention weights for each image
            attn_scores = []
            for i in range(M):
                score = self.image_attention(stacked[:, :, i, :])  # [B, N, 1]
                attn_scores.append(score)
            
            attn_weights = torch.softmax(torch.cat(attn_scores, dim=-1), dim=-1)  # [B, N, num_images]
            
            # Weighted sum
            fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, D]
            
        elif self.fusion_method == 'concat':
            # Concatenate along feature dimension
            concatenated = torch.cat(image_features_list, dim=-1)  # [B, N, D*num_images]
            fused = self.fusion_proj(concatenated)  # [B, N, D]
        
        return fused


class EnhancedMultiLesionCoT(nn.Module):
    """
    Enhanced Medical Multimodal Chain-of-Thought Model
    
    NEW FEATURES:
    1. Multi-lesion segmentation and detection
    2. Multiple image input support
    3. Per-lesion attention and reasoning
    4. Lesion-specific diagnosis + global diagnosis
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
        num_lesion_types: int = 10,
        max_num_images: int = 3,
        image_fusion_method: str = 'attention',
        enable_segmentation: bool = True,
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
            num_lesion_types: Number of lesion types (for segmentation)
            max_num_images: Maximum number of input images
            image_fusion_method: How to fuse multiple images ('attention', 'concat', 'average')
            enable_segmentation: Whether to enable segmentation head
        """
        super().__init__()
        
        self.enable_segmentation = enable_segmentation
        self.max_num_images = max_num_images
        
        # Image encoder (can handle multiple images)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            img_size=img_size,
        )
        
        # Multi-image fusion
        self.image_fusion = MultiImageFusion(
            hidden_dim=self.image_encoder.feature_dim,
            num_images=max_num_images,
            fusion_method=image_fusion_method,
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
        )
        
        # Projection layers
        self.image_proj = nn.Linear(self.image_encoder.feature_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.feature_dim, hidden_dim)
        
        # Segmentation head (optional)
        if enable_segmentation:
            self.segmentation_head = MultiLesionSegmentationHead(
                input_dim=self.image_encoder.feature_dim,
                num_lesion_types=num_lesion_types,
                img_size=img_size,
            )
        
        # RoI feature extraction for per-lesion analysis
        self.lesion_roi_extractor = LesionROIExtractor(
            feature_dim=self.image_encoder.feature_dim,
            output_dim=hidden_dim,
        )
        
        # Cross-modal attention
        self.img_to_text_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.text_to_img_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Per-lesion cross-modal attention
        self.lesion_to_text_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Chain-of-thought decoder
        self.cot_decoder = ChainOfThoughtDecoder(
            dim=hidden_dim,
            num_steps=num_cot_steps,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )
        
        # Region attention (for each CoT step)
        self.region_attention = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for bbox
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Per-lesion diagnosis head
        self.lesion_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # lesion feature + text context
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_lesion_types),
        )
        
        # Per-lesion confidence
        self.lesion_confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Lesion aggregation for global reasoning
        self.lesion_aggregator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Global diagnosis head (considering all lesions)
        self.global_diagnosis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diagnosis_classes),
        )
        
        # Global confidence
        self.global_confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def encode_images(
        self,
        images_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple images and fuse them
        
        Args:
            images_list: List of image tensors, each [B, 3, H, W]
            
        Returns:
            fused_global: [B, feature_dim]
            fused_patches: [B, num_patches, feature_dim]
        """
        if len(images_list) == 1:
            # Single image - no fusion needed
            return self.image_encoder(images_list[0])
        
        # Encode each image
        all_global = []
        all_patches = []
        
        for img in images_list:
            img_global, img_patches = self.image_encoder(img)
            all_global.append(img_global)
            all_patches.append(img_patches)
        
        # Fuse multiple images
        fused_patches = self.image_fusion(all_patches)
        fused_global = torch.stack(all_global, dim=1).mean(dim=1)  # Average global features
        
        return fused_global, fused_patches
    
    def forward(
        self,
        images: torch.Tensor,  # [B, 3, H, W] or [B, num_images, 3, H, W]
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        cot_step_input_ids: torch.Tensor,
        cot_step_attention_mask: torch.Tensor,
        cot_step_regions: torch.Tensor,  # [B, num_steps, 4]
        cot_num_steps: torch.Tensor,
        lesion_bboxes: Optional[torch.Tensor] = None,  # [N, 5]: batch_idx, x1, y1, x2, y2
        lesion_ids: Optional[torch.Tensor] = None,  # [N] which CoT step each lesion corresponds to
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-lesion support
        
        Args:
            images: Medical images
                   - Single image: [B, 3, H, W]
                   - Multiple images: [B, num_images, 3, H, W]
            text_input_ids: Clinical text tokens [B, seq_len]
            text_attention_mask: Text attention mask [B, seq_len]
            cot_step_input_ids: Reasoning step tokens [B, num_steps, seq_len]
            cot_step_attention_mask: Step attention masks [B, num_steps, seq_len]
            cot_step_regions: Bounding boxes [B, num_steps, 4]
            cot_num_steps: Actual number of steps [B]
            lesion_bboxes: Optional lesion locations [N, 5]
            lesion_ids: Optional lesion-to-step mapping [N]
            
        Returns:
            Dictionary containing:
                - segmentation: [B, num_lesion_types+1, H, W] (if enabled)
                - instance_map: [B, 1, H, W] (if enabled)
                - lesion_diagnoses: [N, num_lesion_types] (if lesion_bboxes provided)
                - lesion_confidences: [N, 1] (if lesion_bboxes provided)
                - global_diagnosis_logits: [B, num_classes]
                - global_confidence: [B, 1]
                - step_attentions: [B, num_steps]
                - cross_modal_attention: [B, num_patches, seq_len]
                - per_lesion_attention: [N, num_patches] (if lesion_bboxes provided)
        """
        B = images.size(0)
        
        # Handle multiple images
        if len(images.shape) == 5:  # [B, num_images, 3, H, W]
            num_images = images.size(1)
            images_list = [images[:, i, :, :, :] for i in range(num_images)]
        else:  # [B, 3, H, W]
            images_list = [images]
        
        # Encode and fuse images
        img_global, img_patches = self.encode_images(images_list)
        img_patches_proj = self.image_proj(img_patches)
        
        # Segmentation (if enabled)
        seg_logits = None
        instance_map = None
        if self.enable_segmentation:
            seg_logits, instance_map = self.segmentation_head(img_patches)
        
        # Encode clinical text
        text_pooled, text_sequence = self.text_encoder(
            text_input_ids,
            text_attention_mask,
        )
        text_sequence_proj = self.text_proj(text_sequence)
        
        # Global cross-modal attention
        img_attended, img_to_text_attn = self.img_to_text_attn(
            query=img_patches_proj,
            key=text_sequence_proj,
            value=text_sequence_proj,
            key_padding_mask=~text_attention_mask.bool(),
        )
        
        text_attended, text_to_img_attn = self.text_to_img_attn(
            query=text_sequence_proj,
            key=img_patches_proj,
            value=img_patches_proj,
        )
        
        # Per-lesion analysis (if lesion bboxes provided)
        lesion_diagnoses = None
        lesion_confidences = None
        per_lesion_attention = None
        
        if lesion_bboxes is not None and len(lesion_bboxes) > 0:
            # Reshape patches to spatial map for RoI extraction
            num_patches = img_patches.size(1)
            H = W = int(num_patches ** 0.5)
            img_patches_spatial = img_patches.view(B, H, W, -1)
            
            # Extract per-lesion features
            lesion_features = self.lesion_roi_extractor(
                img_patches_spatial, lesion_bboxes
            )  # [N, hidden_dim]
            
            # Get corresponding text context for each lesion
            # If lesion_ids provided, use step-specific text
            if lesion_ids is not None:
                # Encode CoT steps for these lesions
                text_contexts = []
                for lid in lesion_ids:
                    step_idx = lid.item()
                    if step_idx < cot_step_input_ids.size(1):
                        step_pooled, _ = self.text_encoder(
                            cot_step_input_ids[:, step_idx, :],
                            cot_step_attention_mask[:, step_idx, :],
                        )
                        text_contexts.append(self.text_proj(step_pooled))
                text_context = torch.cat(text_contexts, dim=0)
            else:
                # Use global text context for all lesions
                text_pooled_proj = self.text_proj(text_pooled)
                N = lesion_features.size(0)
                batch_indices = lesion_bboxes[:, 0].long()
                text_context = text_pooled_proj[batch_indices]
            
            # Per-lesion cross-modal reasoning
            lesion_with_text = torch.cat([lesion_features, text_context], dim=-1)
            
            # Per-lesion diagnosis and confidence
            lesion_diagnoses = self.lesion_classifier(lesion_with_text)
            lesion_confidences = self.lesion_confidence(lesion_with_text)
            
            # Per-lesion attention to image patches
            lesion_features_expanded = lesion_features.unsqueeze(1)  # [N, 1, hidden_dim]
            img_patches_for_lesion = img_patches_proj[batch_indices]  # [N, num_patches, hidden_dim]
            
            lesion_attn_output, per_lesion_attention = self.lesion_to_text_attn(
                query=lesion_features_expanded,
                key=img_patches_for_lesion,
                value=img_patches_for_lesion,
            )  # per_lesion_attention: [N, 1, num_patches]
            
            per_lesion_attention = per_lesion_attention.squeeze(1)  # [N, num_patches]
        
        # Combine context for global reasoning
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
        
        # Compute region-specific attention for each step
        regions_normalized = cot_step_regions / images.size(-1)
        region_features = torch.cat([cot_output, regions_normalized], dim=-1)
        region_attn_scores = self.region_attention(region_features).squeeze(-1)  # [B, num_steps]
        
        # Aggregate reasoning (weighted by region attention)
        final_reasoning = (cot_output * F.softmax(region_attn_scores, dim=1).unsqueeze(-1)).sum(dim=1)
        
        # If we have per-lesion features, aggregate them too
        if lesion_bboxes is not None and len(lesion_bboxes) > 0:
            # Group lesion features by batch
            lesion_features_batched = []
            for b in range(B):
                batch_mask = lesion_bboxes[:, 0] == b
                if batch_mask.sum() > 0:
                    lesion_features_batched.append(lesion_features[batch_mask])
                else:
                    # No lesions for this sample - use dummy
                    lesion_features_batched.append(torch.zeros(1, hidden_dim, device=images.device))
            
            # Pad to same length and stack
            max_lesions = max(len(lf) for lf in lesion_features_batched)
            padded_lesions = []
            lesion_masks = []
            
            for lf in lesion_features_batched:
                if len(lf) < max_lesions:
                    padding = torch.zeros(max_lesions - len(lf), hidden_dim, device=images.device)
                    lf_padded = torch.cat([lf, padding], dim=0)
                    mask = torch.cat([
                        torch.ones(len(lf), dtype=torch.bool, device=images.device),
                        torch.zeros(max_lesions - len(lf), dtype=torch.bool, device=images.device)
                    ])
                else:
                    lf_padded = lf
                    mask = torch.ones(len(lf), dtype=torch.bool, device=images.device)
                
                padded_lesions.append(lf_padded)
                lesion_masks.append(mask)
            
            lesion_features_batched = torch.stack(padded_lesions, dim=0)  # [B, max_lesions, hidden_dim]
            lesion_masks = torch.stack(lesion_masks, dim=0)  # [B, max_lesions]
            
            # Aggregate lesion information
            aggregated_lesions, lesion_attn_weights = self.lesion_aggregator(
                query=final_reasoning.unsqueeze(1),  # [B, 1, hidden_dim]
                key=lesion_features_batched,
                value=lesion_features_batched,
                key_padding_mask=~lesion_masks,
            )  # [B, 1, hidden_dim]
            
            # Combine reasoning with lesion information
            final_reasoning = final_reasoning + aggregated_lesions.squeeze(1)
        
        # Global predictions
        global_diagnosis_logits = self.global_diagnosis_classifier(final_reasoning)
        global_confidence = self.global_confidence_predictor(final_reasoning)
        
        outputs = {
            'global_diagnosis_logits': global_diagnosis_logits,
            'global_confidence': global_confidence,
            'step_attentions': region_attn_scores,
            'cross_modal_attention': img_to_text_attn,
            'reasoning_features': cot_output,
        }
        
        # Add segmentation outputs if enabled
        if self.enable_segmentation:
            outputs['segmentation'] = seg_logits
            outputs['instance_map'] = instance_map
        
        # Add per-lesion outputs if available
        if lesion_diagnoses is not None:
            outputs['lesion_diagnoses'] = lesion_diagnoses
            outputs['lesion_confidences'] = lesion_confidences
            outputs['per_lesion_attention'] = per_lesion_attention
        
        return outputs


def create_multi_lesion_model(config: Dict) -> EnhancedMultiLesionCoT:
    """
    Factory function to create enhanced multi-lesion model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized enhanced model
    """
    return EnhancedMultiLesionCoT(**config)


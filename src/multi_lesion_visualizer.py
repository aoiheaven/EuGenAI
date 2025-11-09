"""
Multi-Lesion Visualization Tools

Provides specialized visualizations for:
1. Multi-lesion segmentation overlay
2. Per-lesion attention heatmaps
3. Multiple image comparison
4. Lesion-specific reasoning chains
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MultiLesionVisualizer:
    """Visualization tools for multi-lesion analysis"""
    
    # Color palette for different lesions (up to 10 lesions)
    LESION_COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring green
        (255, 128, 128),  # Pink
    ]
    
    # Severity colors
    SEVERITY_COLORS = {
        'mild': (100, 200, 100),
        'moderate': (255, 165, 0),
        'severe': (255, 50, 50),
    }
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def visualize_multi_lesion_segmentation(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        lesion_info: List[Dict],
        save_path: Optional[str] = None,
    ):
        """
        Visualize multi-lesion segmentation results
        
        Args:
            image: Original image [H, W, 3]
            seg_mask: Segmentation mask [H, W] with lesion IDs
            lesion_info: List of dicts with lesion information
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Medical Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation mask (colored by lesion ID)
        lesion_colored = np.zeros_like(image)
        unique_ids = np.unique(seg_mask)
        
        for lesion_id in unique_ids:
            if lesion_id == 0:  # Skip background
                continue
            mask = seg_mask == lesion_id
            color_idx = (lesion_id - 1) % len(self.LESION_COLORS)
            lesion_colored[mask] = self.LESION_COLORS[color_idx]
        
        axes[0, 1].imshow(lesion_colored)
        axes[0, 1].set_title('Segmentation Mask (By Lesion)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, lesion_colored, 0.4, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Detection with bboxes and labels
        img_with_boxes = image.copy()
        
        for i, lesion in enumerate(lesion_info):
            if i >= len(self.LESION_COLORS):
                break
            
            bbox = lesion.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 5:  # Has batch_idx
                bbox = bbox[1:]
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.LESION_COLORS[i]
            
            # Draw bbox
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            lesion_type = lesion.get('type', 'Unknown')
            confidence = lesion.get('confidence', 0.0)
            label_text = f"L{i+1}: {lesion_type} ({confidence:.2f})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img_with_boxes, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
            cv2.putText(
                img_with_boxes,
                label_text,
                (x1+5, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        axes[1, 1].imshow(img_with_boxes)
        axes[1, 1].set_title(f'Lesion Detection (N={len(lesion_info)})', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=np.array(self.LESION_COLORS[i])/255, 
                          label=f"Lesion {i+1}: {lesion_info[i].get('type', 'Unknown')}")
            for i in range(min(len(lesion_info), len(self.LESION_COLORS)))
        ]
        axes[1, 1].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.suptitle('Multi-Lesion Segmentation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_per_lesion_attention(
        self,
        image: np.ndarray,
        lesion_info: List[Dict],
        attention_maps: List[np.ndarray],
        save_path: Optional[str] = None,
    ):
        """
        Visualize attention map for each lesion separately
        
        Args:
            image: Original image [H, W, 3]
            lesion_info: List of lesion dictionaries
            attention_maps: List of attention maps [H, W] for each lesion
            save_path: Path to save
        """
        num_lesions = len(lesion_info)
        cols = min(4, num_lesions)
        rows = (num_lesions + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if num_lesions == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (lesion, attention) in enumerate(zip(lesion_info, attention_maps)):
            # Resize attention to match image
            if len(attention.shape) == 1:
                side_len = int(np.sqrt(len(attention)))
                attention = attention.reshape(side_len, side_len)
            
            attention_resized = cv2.resize(attention, (image.shape[1], image.shape[0]))
            
            # Normalize
            attention_norm = (attention_resized - attention_resized.min()) / \
                           (attention_resized.max() - attention_resized.min() + 1e-8)
            
            # Create heatmap
            heatmap = cv2.applyColorMap((attention_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
            
            # Draw bbox
            bbox = lesion.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 5:
                bbox = bbox[1:]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            
            axes[i].imshow(overlay)
            
            lesion_type = lesion.get('type', 'Unknown')
            confidence = lesion.get('confidence', 0.0)
            max_attention = attention_norm.max()
            
            axes[i].set_title(
                f"Lesion {i+1}: {lesion_type}\n"
                f"Confidence: {confidence:.2f} | Max Attn: {max_attention:.2f}",
                fontsize=11,
                fontweight='bold'
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_lesions, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Per-Lesion Attention Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_multi_image_comparison(
        self,
        images: List[np.ndarray],
        image_labels: List[str],
        attention_maps: Optional[List[np.ndarray]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize multiple images with their attention maps
        
        Useful for temporal analysis or multi-sequence MRI
        
        Args:
            images: List of images
            image_labels: Labels for each image (e.g., 'T1', 'T2', 'Pre-contrast')
            attention_maps: Optional attention maps for each image
            save_path: Path to save
        """
        num_images = len(images)
        
        if attention_maps:
            fig, axes = plt.subplots(2, num_images, figsize=(6*num_images, 12))
        else:
            fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 6))
            axes = axes.reshape(1, -1)
        
        for i, (img, label) in enumerate(zip(images, image_labels)):
            # Original image
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'{label}\n(Original)', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Attention overlay
            if attention_maps and i < len(attention_maps):
                attn = attention_maps[i]
                
                # Resize and normalize
                if len(attn.shape) == 1:
                    side_len = int(np.sqrt(len(attn)))
                    attn = attn.reshape(side_len, side_len)
                
                attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
                attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
                
                heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{label}\n(With Attention)', fontsize=12, fontweight='bold')
                axes[1, i].axis('off')
        
        plt.suptitle('Multi-Image Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_lesion_reasoning_chain(
        self,
        image: np.ndarray,
        lesion_info: Dict,
        reasoning_steps: List[Dict],
        save_path: Optional[str] = None,
    ):
        """
        Visualize reasoning chain for a specific lesion
        
        Args:
            image: Medical image
            lesion_info: Information about the specific lesion
            reasoning_steps: Steps related to this lesion
            save_path: Save path
        """
        num_steps = len(reasoning_steps)
        
        fig, axes = plt.subplots(1, num_steps + 1, figsize=(5*(num_steps+1), 5))
        
        # Show lesion location
        img_with_lesion = image.copy()
        bbox = lesion_info.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 5:
            bbox = bbox[1:]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_with_lesion, (x1, y1), (x2, y2), (255, 255, 0), 4)
        
        axes[0].imshow(img_with_lesion)
        lesion_type = lesion_info.get('type', 'Unknown')
        axes[0].set_title(f"Lesion: {lesion_type}\nLocation", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Show reasoning steps
        for i, step in enumerate(reasoning_steps):
            img_copy = image.copy()
            
            # Highlight region for this step
            step_bbox = step.get('region_of_interest', {}).get('bbox', bbox)
            sx1, sy1, sx2, sy2 = map(int, step_bbox)
            cv2.rectangle(img_copy, (sx1, sy1), (sx2, sy2), (0, 255, 255), 3)
            
            axes[i+1].imshow(img_copy)
            axes[i+1].set_title(
                f"Step {i+1}: {step.get('action', '')}\n"
                f"Attn: {step.get('attention', 0):.2f}",
                fontsize=11,
                fontweight='bold'
            )
            axes[i+1].axis('off')
            
            # Add observation
            obs = step.get('observation', '')[:80]
            axes[i+1].text(
                0.5, -0.1,
                obs,
                transform=axes[i+1].transAxes,
                ha='center',
                fontsize=9,
                style='italic'
            )
        
        plt.suptitle(f'Reasoning Chain for Lesion: {lesion_type}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comprehensive_multi_lesion_report(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        lesion_info: List[Dict],
        global_attention: np.ndarray,
        per_lesion_attention: List[np.ndarray],
        diagnosis: Dict,
        save_dir: str,
    ):
        """
        Generate comprehensive visualization report for multi-lesion case
        
        Args:
            image: Original image
            seg_mask: Segmentation mask
            lesion_info: List of lesion dictionaries
            global_attention: Global attention map
            per_lesion_attention: List of per-lesion attention maps
            diagnosis: Diagnosis information
            save_dir: Directory to save all visualizations
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comprehensive report in {save_dir}/...")
        
        # 1. Segmentation overview
        self.visualize_multi_lesion_segmentation(
            image, seg_mask, lesion_info,
            save_path=str(save_path / '1_segmentation_overview.png')
        )
        
        # 2. Per-lesion attention
        self.visualize_per_lesion_attention(
            image, lesion_info, per_lesion_attention,
            save_path=str(save_path / '2_per_lesion_attention.png')
        )
        
        # 3. Summary figure
        self._create_summary_figure(
            image, lesion_info, diagnosis,
            save_path=str(save_path / '3_summary.png')
        )
        
        print(f"✓ Report generated successfully!")
    
    def _create_summary_figure(
        self,
        image: np.ndarray,
        lesion_info: List[Dict],
        diagnosis: Dict,
        save_path: str,
    ):
        """Create summary figure with key information"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main image with all lesions
        ax_main = fig.add_subplot(gs[0, :])
        img_annotated = image.copy()
        
        for i, lesion in enumerate(lesion_info):
            bbox = lesion.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 5:
                bbox = bbox[1:]
            x1, y1, x2, y2 = map(int, bbox)
            
            color = self.LESION_COLORS[i % len(self.LESION_COLORS)]
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
            
            # Add number marker
            cv2.circle(img_annotated, (x1+20, y1+20), 20, color, -1)
            cv2.putText(img_annotated, str(i+1), (x1+12, y1+28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ax_main.imshow(img_annotated)
        ax_main.set_title('Detected Lesions', fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Lesion details table
        ax_table = fig.add_subplot(gs[1, 0])
        ax_table.axis('off')
        
        table_data = []
        for i, lesion in enumerate(lesion_info):
            table_data.append([
                i+1,
                lesion.get('type', 'Unknown'),
                lesion.get('severity', 'N/A'),
                f"{lesion.get('confidence', 0):.2f}",
            ])
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=['#', 'Type', 'Severity', 'Confidence'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax_table.set_title('Lesion Details', fontsize=14, fontweight='bold')
        
        # Final diagnosis
        ax_diagnosis = fig.add_subplot(gs[1, 1])
        ax_diagnosis.axis('off')
        
        diag_text = f"Primary Diagnosis:\n{diagnosis.get('primary', 'Unknown')}\n\n"
        
        if 'secondary' in diagnosis and diagnosis['secondary']:
            diag_text += "Secondary Findings:\n"
            for sec in diagnosis['secondary']:
                diag_text += f"• {sec}\n"
        
        diag_text += f"\nGlobal Confidence: {diagnosis.get('confidence', 0):.2%}\n"
        diag_text += f"Urgency: {diagnosis.get('urgency', 'N/A')}"
        
        ax_diagnosis.text(
            0.5, 0.5,
            diag_text,
            transform=ax_diagnosis.transAxes,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8)
        )
        ax_diagnosis.set_title('Final Diagnosis', fontsize=14, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def visualize_multi_lesion_comparison(
    ground_truth_masks: List[np.ndarray],
    predicted_masks: List[np.ndarray],
    lesion_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Compare ground truth and predicted segmentation for multiple lesions
    
    Args:
        ground_truth_masks: List of GT masks
        predicted_masks: List of predicted masks
        lesion_names: Names of lesions
        save_path: Save path
    """
    num_lesions = len(ground_truth_masks)
    
    fig, axes = plt.subplots(num_lesions, 3, figsize=(15, 5*num_lesions))
    if num_lesions == 1:
        axes = axes.reshape(1, -1)
    
    for i, (gt, pred, name) in enumerate(zip(ground_truth_masks, predicted_masks, lesion_names)):
        # Ground truth
        axes[i, 0].imshow(gt, cmap='Reds', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{name}\nGround Truth', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{name}\nPrediction', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Difference
        diff = np.abs(gt.astype(float) - pred.astype(float))
        axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'{name}\nDifference', fontweight='bold')
        axes[i, 2].axis('off')
        
        # Compute Dice score
        intersection = (gt * pred).sum()
        dice = 2 * intersection / (gt.sum() + pred.sum() + 1e-8)
        axes[i, 2].text(
            0.5, -0.05,
            f'Dice: {dice:.3f}',
            transform=axes[i, 2].transAxes,
            ha='center',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.suptitle('Multi-Lesion Segmentation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


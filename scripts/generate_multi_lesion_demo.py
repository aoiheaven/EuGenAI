#!/usr/bin/env python3
"""
Generate Multi-Lesion Demo Visualizations

Creates comprehensive demonstrations for:
1. Multi-lesion segmentation
2. Multi-image fusion analysis
3. Per-lesion attention maps
4. Multi-attention-point visualization
5. Lesion comparison dashboard
6. Comprehensive multi-lesion report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path("demo_multi_lesion_visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Generating Multi-Lesion & Multi-Image Demo Visualizations")
print("=" * 80)

# Color schemes
LESION_COLORS = [
    (255, 0, 0),      # Lesion 1: Red
    (0, 255, 0),      # Lesion 2: Green
    (0, 0, 255),      # Lesion 3: Blue
    (255, 255, 0),    # Lesion 4: Yellow
    (255, 0, 255),    # Lesion 5: Magenta
]

SEVERITY_COLORS = {
    'mild': (100, 200, 100),
    'moderate': (255, 165, 0),
    'severe': (255, 50, 50),
}

# ============================================================================
# Helper Functions
# ============================================================================

def create_synthetic_medical_image(size=512, seed=42):
    """Create synthetic medical image"""
    np.random.seed(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [25, 25, 35]
    
    # Simulated organs
    cv2.ellipse(img, (200, 280), (110, 140), 0, 0, 360, (45, 45, 55), -1)
    cv2.ellipse(img, (312, 280), (110, 140), 0, 0, 360, (45, 45, 55), -1)
    
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def create_lesion_mask(size=512, center=(256, 256), radius=30, irregular=True):
    """Create synthetic lesion mask"""
    mask = np.zeros((size, size), dtype=np.uint8)
    
    if irregular:
        # Irregular shape (more realistic)
        points = []
        for angle in np.linspace(0, 2*np.pi, 20):
            r = radius * (1 + 0.3 * np.random.randn())
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    else:
        # Smooth circular
        cv2.circle(mask, center, radius, 1, -1)
    
    return mask

def create_attention_for_lesion(size=512, center=(256, 256), strength=1.0):
    """Create attention map focused on lesion"""
    y, x = np.ogrid[:size, :size]
    cx, cy = center
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    attention = np.exp(-dist**2 / (2 * (40*strength)**2))
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    return attention

# ============================================================================
# 1. Multi-Lesion Segmentation Overview
# ============================================================================
print("\n[1/7] Generating Multi-Lesion Segmentation Overview...")

# Create base image
base_image = create_synthetic_medical_image()

# Create 3 lesions
lesions_info = [
    {'id': 1, 'center': (280, 220), 'radius': 35, 'type': 'Nodule', 'severity': 'severe', 'conf': 0.92},
    {'id': 2, 'center': (180, 300), 'radius': 25, 'type': 'Mass', 'severity': 'moderate', 'conf': 0.85},
    {'id': 3, 'center': (350, 320), 'radius': 20, 'type': 'Opacity', 'severity': 'mild', 'conf': 0.78},
]

# Create segmentation masks
combined_mask = np.zeros((512, 512), dtype=np.uint8)
individual_masks = []

for lesion in lesions_info:
    mask = create_lesion_mask(512, lesion['center'], lesion['radius'])
    combined_mask[mask > 0] = lesion['id']
    individual_masks.append(mask)

# Visualization
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

# Original image
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(base_image)
ax1.set_title('Original Medical Image\n(Simulated CT Scan)', fontsize=13, fontweight='bold')
ax1.axis('off')

# Segmentation mask (colored)
ax2 = fig.add_subplot(gs[0, 1])
seg_colored = np.zeros((512, 512, 3), dtype=np.uint8)
for i, lesion in enumerate(lesions_info):
    seg_colored[combined_mask == lesion['id']] = LESION_COLORS[i]

ax2.imshow(seg_colored)
ax2.set_title('Segmentation Mask\n(3 Detected Lesions)', fontsize=13, fontweight='bold')
ax2.axis('off')

# Overlay
ax3 = fig.add_subplot(gs[0, 2])
overlay = cv2.addWeighted(base_image, 0.65, seg_colored, 0.35, 0)
ax3.imshow(overlay)
ax3.set_title('Segmentation Overlay\n(Combined View)', fontsize=13, fontweight='bold')
ax3.axis('off')

# Detection with bboxes
ax4 = fig.add_subplot(gs[0, 3])
img_boxes = base_image.copy()
for i, lesion in enumerate(lesions_info):
    cx, cy = lesion['center']
    r = lesion['radius']
    x1, y1, x2, y2 = cx-r-10, cy-r-10, cx+r+10, cy+r+10
    
    color = LESION_COLORS[i]
    cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 3)
    
    # Label
    label = f"L{i+1}: {lesion['type']}\n{lesion['conf']:.2f}"
    cv2.circle(img_boxes, (x1+15, y1+15), 15, color, -1)
    cv2.putText(img_boxes, str(i+1), (x1+8, y1+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

ax4.imshow(img_boxes)
ax4.set_title('Lesion Detection & Classification\n(Bounding Boxes)', fontsize=13, fontweight='bold')
ax4.axis('off')

# Individual lesion masks
for i, (lesion, mask) in enumerate(zip(lesions_info, individual_masks)):
    ax = fig.add_subplot(gs[1, i])
    
    # Show lesion on image
    lesion_overlay = base_image.copy()
    lesion_colored = np.zeros_like(base_image)
    lesion_colored[mask > 0] = LESION_COLORS[i]
    lesion_overlay = cv2.addWeighted(lesion_overlay, 0.7, lesion_colored, 0.3, 0)
    
    ax.imshow(lesion_overlay)
    ax.set_title(f"Lesion {i+1}: {lesion['type']}\n"
                f"Severity: {lesion['severity']} | Conf: {lesion['conf']:.2f}",
                fontsize=11, fontweight='bold')
    ax.axis('off')

# Summary statistics
ax_summary = fig.add_subplot(gs[1, 3])
ax_summary.axis('off')

summary_text = "📊 Detection Summary\n\n"
summary_text += f"Total Lesions: {len(lesions_info)}\n\n"
summary_text += "Lesion Details:\n"
for i, lesion in enumerate(lesions_info):
    summary_text += f"• L{i+1}: {lesion['type']}\n"
    summary_text += f"  Size: ~{lesion['radius']*2}px\n"
    summary_text += f"  Risk: {lesion['severity']}\n"
    summary_text += f"  Conf: {lesion['conf']:.2%}\n\n"

ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
               fontsize=11, ha='center', va='center', family='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Multi-Lesion Segmentation & Detection System', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '1_multi_lesion_segmentation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '1_multi_lesion_segmentation.png'}")
plt.close()

# ============================================================================
# 2. Multi-Image Fusion (MRI Multi-Sequence)
# ============================================================================
print("\n[2/7] Generating Multi-Image Fusion Visualization...")

# Create 3 different "sequences"
def create_mri_sequence(base, sequence_type):
    """Simulate different MRI sequences"""
    img = base.copy()
    
    if sequence_type == 'T1':
        # T1: Good anatomical detail
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    elif sequence_type == 'T2':
        # T2: Bright fluid/edema
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=20)
        # Add "edema" around lesions
        for lesion in lesions_info[:2]:
            cv2.circle(img, lesion['center'], lesion['radius']+20, (80, 80, 90), -1)
            cv2.circle(img, lesion['center'], lesion['radius'], (100, 100, 110), -1)
    elif sequence_type == 'FLAIR':
        # FLAIR: Suppress CSF, bright lesions
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=-10)
        for lesion in lesions_info:
            cv2.circle(img, lesion['center'], lesion['radius'], (120, 120, 130), -1)
    
    return img

t1_image = create_mri_sequence(base_image, 'T1')
t2_image = create_mri_sequence(base_image, 'T2')
flair_image = create_mri_sequence(base_image, 'FLAIR')

# Create attention weights for each sequence (learned by model)
sequence_weights = [0.45, 0.35, 0.20]  # T1 most important for this case

# Create per-sequence attention maps
t1_attention = create_attention_for_lesion(512, lesions_info[0]['center'], 1.2)
t2_attention = create_attention_for_lesion(512, lesions_info[1]['center'], 0.9)
flair_attention = create_attention_for_lesion(512, lesions_info[0]['center'], 0.7)

# Visualize
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)

sequences = [
    ('T1-weighted', t1_image, t1_attention, sequence_weights[0]),
    ('T2-weighted', t2_image, t2_attention, sequence_weights[1]),
    ('FLAIR', flair_image, flair_attention, sequence_weights[2]),
]

for i, (seq_name, img, attn, weight) in enumerate(sequences):
    # Original image
    ax_orig = fig.add_subplot(gs[0, i])
    ax_orig.imshow(img)
    ax_orig.set_title(f'{seq_name}\nFusion Weight: {weight:.1%}', 
                     fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # Attention map
    ax_attn = fig.add_subplot(gs[1, i])
    im = ax_attn.imshow(attn, cmap='hot', vmin=0, vmax=1)
    ax_attn.set_title(f'Attention Map\nMax: {attn.max():.2f}', 
                     fontsize=11, fontweight='bold')
    ax_attn.axis('off')
    plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)
    
    # Overlay
    ax_overlay = fig.add_subplot(gs[2, i])
    heatmap = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    ax_overlay.imshow(overlay_img)
    ax_overlay.set_title(f'Attention Overlay', fontsize=11, fontweight='bold')
    ax_overlay.axis('off')

# Fused result
ax_fused = fig.add_subplot(gs[:, 3])

# Weighted fusion
fused_image = (t1_image.astype(float) * sequence_weights[0] + 
               t2_image.astype(float) * sequence_weights[1] + 
               flair_image.astype(float) * sequence_weights[2])
fused_image = fused_image.astype(np.uint8)

fused_attention = (t1_attention * sequence_weights[0] + 
                   t2_attention * sequence_weights[1] + 
                   flair_attention * sequence_weights[2])

fused_heatmap = cv2.applyColorMap((fused_attention * 255).astype(np.uint8), cv2.COLORMAP_JET)
fused_heatmap = cv2.cvtColor(fused_heatmap, cv2.COLOR_BGR2RGB)
fused_overlay = cv2.addWeighted(fused_image, 0.6, fused_heatmap, 0.4, 0)

ax_fused.imshow(fused_overlay)
ax_fused.set_title('Fused Multi-Sequence Result\n(Attention-Weighted Fusion)', 
                  fontsize=14, fontweight='bold')
ax_fused.axis('off')

# Add fusion formula
formula_text = "Fusion Formula:\n"
formula_text += "F = 0.45×T1 + 0.35×T2 + 0.20×FLAIR\n\n"
formula_text += "Learned Weights:\n"
formula_text += "• T1: Best anatomical detail\n"
formula_text += "• T2: Shows edema clearly\n"
formula_text += "• FLAIR: Lesion contrast"

ax_fused.text(0.5, -0.15, formula_text, transform=ax_fused.transAxes,
             fontsize=10, ha='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7))

plt.suptitle('Multi-Image Fusion Analysis (MRI Multi-Sequence)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '2_multi_image_fusion.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '2_multi_image_fusion.png'}")
plt.close()

# ============================================================================
# 3. Per-Lesion Attention Maps
# ============================================================================
print("\n[3/7] Generating Per-Lesion Attention Maps...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, lesion in enumerate(lesions_info):
    # Create lesion-specific attention
    lesion_attention = create_attention_for_lesion(
        512, lesion['center'], strength=1.5
    )
    
    # Add some context attention
    for other in lesions_info:
        if other['id'] != lesion['id']:
            context_attn = create_attention_for_lesion(
                512, other['center'], strength=0.3
            )
            lesion_attention = np.maximum(lesion_attention, context_attn * 0.2)
    
    # Original with bbox
    ax_img = axes[0, i]
    img_copy = base_image.copy()
    cx, cy = lesion['center']
    r = lesion['radius']
    cv2.rectangle(img_copy, (cx-r-10, cy-r-10), (cx+r+10, cy+r+10), 
                 LESION_COLORS[i], 3)
    ax_img.imshow(img_copy)
    ax_img.set_title(f"Lesion {i+1}: {lesion['type']}", 
                    fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Attention heatmap
    ax_attn = axes[1, i]
    heatmap = cv2.applyColorMap((lesion_attention * 255).astype(np.uint8), 
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_image, 0.6, heatmap, 0.4, 0)
    
    # Draw circle to highlight lesion
    cv2.circle(overlay, lesion['center'], lesion['radius']+5, (255, 255, 0), 2)
    
    ax_attn.imshow(overlay)
    
    max_attn = lesion_attention.max()
    mean_attn = lesion_attention[individual_masks[i] > 0].mean()
    
    ax_attn.set_title(f"Attention Map\n"
                     f"Max: {max_attn:.2f} | Mean (in lesion): {mean_attn:.2f}",
                     fontsize=11, fontweight='bold')
    ax_attn.axis('off')
    
    # Add confidence bar
    severity_color = SEVERITY_COLORS[lesion['severity']]
    severity_color_norm = tuple(c/255 for c in severity_color)
    
    confidence_text = f"Confidence: {lesion['conf']:.0%}\nSeverity: {lesion['severity']}"
    ax_attn.text(0.5, -0.12, confidence_text, transform=ax_attn.transAxes,
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor=severity_color_norm, alpha=0.6))

plt.suptitle('Per-Lesion Independent Attention Analysis', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '3_per_lesion_attention.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '3_per_lesion_attention.png'}")
plt.close()

# ============================================================================
# 4. Multi-Attention-Point Comparison
# ============================================================================
print("\n[4/7] Generating Multi-Attention-Point Comparison...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Global attention (all lesions)
global_attention = np.zeros((512, 512))
for lesion in lesions_info:
    global_attention += create_attention_for_lesion(512, lesion['center'], 1.0)
global_attention = global_attention / global_attention.max()

ax_global = fig.add_subplot(gs[0, :])
global_heatmap = cv2.applyColorMap((global_attention * 255).astype(np.uint8), 
                                   cv2.COLORMAP_JET)
global_heatmap = cv2.cvtColor(global_heatmap, cv2.COLOR_BGR2RGB)
global_overlay = cv2.addWeighted(base_image, 0.6, global_heatmap, 0.4, 0)
ax_global.imshow(global_overlay)
ax_global.set_title('Level 1: Global Attention (All Lesions)', 
                   fontsize=14, fontweight='bold', color='blue')
ax_global.axis('off')

# Per-lesion attention (3 lesions)
for i, lesion in enumerate(lesions_info):
    ax = fig.add_subplot(gs[1, i])
    
    lesion_attn = create_attention_for_lesion(512, lesion['center'], 1.3)
    lesion_heat = cv2.applyColorMap((lesion_attn * 255).astype(np.uint8), 
                                   cv2.COLORMAP_JET)
    lesion_heat = cv2.cvtColor(lesion_heat, cv2.COLOR_BGR2RGB)
    lesion_over = cv2.addWeighted(base_image, 0.6, lesion_heat, 0.4, 0)
    
    ax.imshow(lesion_over)
    ax.set_title(f'Level 2: Lesion {i+1} Attention\n({lesion["type"]})', 
                fontsize=12, fontweight='bold', color='green')
    ax.axis('off')

# Step-wise attention (reasoning chain)
reasoning_steps = [
    {'name': 'Survey', 'regions': [0, 1, 2], 'weight': 0.65},
    {'name': 'Focus L1', 'regions': [0], 'weight': 0.95},
    {'name': 'Focus L2', 'regions': [1], 'weight': 0.88},
]

for i, step in enumerate(reasoning_steps):
    ax = fig.add_subplot(gs[2, i])
    
    # Combine attention for regions in this step
    step_attn = np.zeros((512, 512))
    for region_id in step['regions']:
        if region_id < len(lesions_info):
            step_attn += create_attention_for_lesion(
                512, lesions_info[region_id]['center'], 0.8
            )
    step_attn = step_attn / (step_attn.max() + 1e-8)
    
    step_heat = cv2.applyColorMap((step_attn * 255).astype(np.uint8), 
                                 cv2.COLORMAP_JET)
    step_heat = cv2.cvtColor(step_heat, cv2.COLOR_BGR2RGB)
    step_over = cv2.addWeighted(base_image, 0.6, step_heat, 0.4, 0)
    
    ax.imshow(step_over)
    ax.set_title(f'Level 3: Step {i+1} - {step["name"]}\n'
                f'Attention Weight: {step["weight"]:.2f}',
                fontsize=11, fontweight='bold', color='red')
    ax.axis('off')

plt.suptitle('Multi-Level Attention Hierarchy (3 Levels)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '4_multi_attention_levels.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '4_multi_attention_levels.png'}")
plt.close()

# ============================================================================
# 5. Lesion-Specific Reasoning Chains
# ============================================================================
print("\n[5/7] Generating Lesion-Specific Reasoning Chains...")

fig, axes = plt.subplots(3, 5, figsize=(24, 14))

lesion_reasoning = [
    {
        'lesion_id': 1,
        'type': 'Nodule',
        'steps': [
            {'action': 'Locate', 'obs': 'Right upper lobe', 'bbox_expand': 1.0, 'attn': 0.72},
            {'action': 'Examine shape', 'obs': 'Spiculated margin', 'bbox_expand': 1.2, 'attn': 0.89},
            {'action': 'Assess density', 'obs': 'Solid component', 'bbox_expand': 1.0, 'attn': 0.91},
            {'action': 'Correlate size', 'obs': '18mm diameter', 'bbox_expand': 1.1, 'attn': 0.85},
            {'action': 'Conclude', 'obs': 'High malignancy risk', 'bbox_expand': 1.0, 'attn': 0.95},
        ]
    },
    {
        'lesion_id': 2,
        'type': 'Mass',
        'steps': [
            {'action': 'Locate', 'obs': 'Left upper lobe', 'bbox_expand': 1.0, 'attn': 0.68},
            {'action': 'Examine margins', 'obs': 'Irregular borders', 'bbox_expand': 1.2, 'attn': 0.82},
            {'action': 'Check enhancement', 'obs': 'Heterogeneous', 'bbox_expand': 1.0, 'attn': 0.87},
            {'action': 'Assess necrosis', 'obs': 'Central lucency', 'bbox_expand': 1.1, 'attn': 0.79},
            {'action': 'Conclude', 'obs': 'Moderate risk', 'bbox_expand': 1.0, 'attn': 0.85},
        ]
    },
    {
        'lesion_id': 3,
        'type': 'Opacity',
        'steps': [
            {'action': 'Locate', 'obs': 'Right lower lobe', 'bbox_expand': 1.0, 'attn': 0.61},
            {'action': 'Examine pattern', 'obs': 'Ground glass', 'bbox_expand': 1.2, 'attn': 0.74},
            {'action': 'Check bronchi', 'obs': 'Air bronchogram', 'bbox_expand': 1.0, 'attn': 0.79},
            {'action': 'Assess extent', 'obs': 'Localized', 'bbox_expand': 1.1, 'attn': 0.72},
            {'action': 'Conclude', 'obs': 'Likely inflammatory', 'bbox_expand': 1.0, 'attn': 0.78},
        ]
    }
]

for lesion_idx, reasoning in enumerate(lesion_reasoning):
    lesion = lesions_info[lesion_idx]
    
    for step_idx, step in enumerate(reasoning['steps']):
        ax = axes[lesion_idx, step_idx]
        
        img_copy = base_image.copy()
        
        # Draw bbox
        cx, cy = lesion['center']
        r = int(lesion['radius'] * step['bbox_expand'])
        x1, y1, x2, y2 = cx-r-10, cy-r-10, cx+r+10, cy+r+10
        
        color = LESION_COLORS[lesion_idx]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        # Highlight center
        cv2.circle(img_copy, (cx, cy), 5, (255, 255, 0), -1)
        
        ax.imshow(img_copy)
        ax.set_title(f"L{lesion_idx+1}-S{step_idx+1}: {step['action']}\n"
                    f"Attn: {step['attn']:.2f}",
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add observation
        ax.text(0.5, -0.08, step['obs'],
               transform=ax.transAxes, ha='center', fontsize=9,
               style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

# Add row labels
for i, reasoning in enumerate(lesion_reasoning):
    axes[i, 0].text(-0.3, 0.5, f"Lesion {i+1}\n{reasoning['type']}", 
                   transform=axes[i, 0].transAxes,
                   fontsize=13, fontweight='bold', rotation=90,
                   va='center', ha='center',
                   bbox=dict(boxstyle='round,pad=0.8', 
                            facecolor=tuple(c/255 for c in LESION_COLORS[i]), 
                            alpha=0.3))

plt.suptitle('Lesion-Specific Reasoning Chains (3 Lesions × 5 Steps)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '5_lesion_reasoning_chains.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '5_lesion_reasoning_chains.png'}")
plt.close()

# ============================================================================
# 6. Multi-Lesion Comparison Dashboard
# ============================================================================
print("\n[6/7] Generating Multi-Lesion Comparison Dashboard...")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

# Top: Main image with all lesions annotated
ax_main = fig.add_subplot(gs[0, :])
img_annotated = base_image.copy()

for i, lesion in enumerate(lesions_info):
    cx, cy = lesion['center']
    r = lesion['radius']
    color = LESION_COLORS[i]
    
    # Draw bbox
    cv2.rectangle(img_annotated, (cx-r-10, cy-r-10), (cx+r+10, cy+r+10), color, 3)
    
    # Number marker
    cv2.circle(img_annotated, (cx-r-5, cy-r-5), 18, color, -1)
    cv2.putText(img_annotated, str(i+1), (cx-r-12, cy-r), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Severity indicator
    severity_color = SEVERITY_COLORS[lesion['severity']]
    cv2.circle(img_annotated, (cx+r+5, cy+r+5), 8, severity_color, -1)

ax_main.imshow(img_annotated)
ax_main.set_title('Detected Lesions Overview (N=3)', fontsize=14, fontweight='bold')
ax_main.axis('off')

# Lesion comparison table
ax_table = fig.add_subplot(gs[1, :2])
ax_table.axis('off')

table_data = []
for i, lesion in enumerate(lesions_info):
    size = lesion['radius'] * 2
    table_data.append([
        f"L{i+1}",
        lesion['type'],
        f"{size}px",
        lesion['severity'].capitalize(),
        f"{lesion['conf']:.1%}",
        "High" if lesion['conf'] > 0.85 else "Moderate"
    ])

table = ax_table.table(
    cellText=table_data,
    colLabels=['ID', 'Type', 'Size', 'Severity', 'Confidence', 'Priority'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
    colWidths=[0.1, 0.2, 0.15, 0.2, 0.2, 0.15]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows by severity
severity_row_colors = {
    'severe': '#ffcccc',
    'moderate': '#ffffcc',
    'mild': '#ccffcc',
}

for i, lesion in enumerate(lesions_info):
    row_color = severity_row_colors[lesion['severity']]
    for j in range(6):
        table[(i+1, j)].set_facecolor(row_color)

ax_table.set_title('Lesion Comparison Table', fontsize=13, fontweight='bold')

# Confidence distribution
ax_conf = fig.add_subplot(gs[1, 2:])
confidences = [l['conf'] for l in lesions_info]
lesion_names = [f"L{i+1}\n{l['type']}" for i, l in enumerate(lesions_info)]
colors_conf = [tuple(c/255 for c in LESION_COLORS[i]) for i in range(len(lesions_info))]

bars = ax_conf.barh(lesion_names, confidences, color=colors_conf, alpha=0.7, edgecolor='black')
ax_conf.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
ax_conf.set_title('Per-Lesion Confidence Scores', fontsize=13, fontweight='bold')
ax_conf.set_xlim(0, 1)
ax_conf.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, conf) in enumerate(zip(bars, confidences)):
    ax_conf.text(conf + 0.02, i, f'{conf:.1%}', 
                va='center', fontsize=11, fontweight='bold')

# Attention intensity comparison
ax_attn_comp = fig.add_subplot(gs[2, :2])

attention_stats = []
for i, lesion in enumerate(lesions_info):
    lesion_attn = create_attention_for_lesion(512, lesion['center'], 1.0)
    mask = individual_masks[i]
    
    max_attn = lesion_attn.max()
    mean_attn = lesion_attn[mask > 0].mean()
    std_attn = lesion_attn[mask > 0].std()
    
    attention_stats.append({
        'name': f"L{i+1}",
        'max': max_attn,
        'mean': mean_attn,
        'std': std_attn
    })

x = np.arange(len(attention_stats))
width = 0.25

bars1 = ax_attn_comp.bar(x - width, [s['max'] for s in attention_stats], 
                        width, label='Max', alpha=0.8, color='#e74c3c')
bars2 = ax_attn_comp.bar(x, [s['mean'] for s in attention_stats], 
                        width, label='Mean', alpha=0.8, color='#3498db')
bars3 = ax_attn_comp.bar(x + width, [s['std'] for s in attention_stats], 
                        width, label='Std Dev', alpha=0.8, color='#2ecc71')

ax_attn_comp.set_xlabel('Lesion', fontsize=12, fontweight='bold')
ax_attn_comp.set_ylabel('Attention Intensity', fontsize=12, fontweight='bold')
ax_attn_comp.set_title('Attention Statistics per Lesion', fontsize=13, fontweight='bold')
ax_attn_comp.set_xticks(x)
ax_attn_comp.set_xticklabels([s['name'] for s in attention_stats])
ax_attn_comp.legend(fontsize=11)
ax_attn_comp.grid(axis='y', alpha=0.3)
ax_attn_comp.set_ylim(0, 1)

# Size vs Confidence scatter
ax_scatter = fig.add_subplot(gs[2, 2:])

sizes = [l['radius'] * 2 for l in lesions_info]
confs = [l['conf'] for l in lesions_info]
colors_scatter = [tuple(c/255 for c in LESION_COLORS[i]) for i in range(len(lesions_info))]

ax_scatter.scatter(sizes, confs, s=500, c=colors_scatter, alpha=0.6, 
                  edgecolors='black', linewidth=2)

# Add labels
for i, (size, conf, lesion) in enumerate(zip(sizes, confs, lesions_info)):
    ax_scatter.annotate(f"L{i+1}\n{lesion['type']}", 
                       (size, conf), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=colors_scatter[i], alpha=0.3),
                       arrowprops=dict(arrowstyle='->', lw=1.5))

ax_scatter.set_xlabel('Lesion Size (pixels)', fontsize=12, fontweight='bold')
ax_scatter.set_ylabel('Diagnostic Confidence', fontsize=12, fontweight='bold')
ax_scatter.set_title('Size vs Confidence Relationship', fontsize=13, fontweight='bold')
ax_scatter.grid(True, alpha=0.3)
ax_scatter.set_xlim(0, max(sizes) * 1.3)
ax_scatter.set_ylim(0.7, 1.0)

plt.suptitle('Multi-Lesion Comparative Analysis Dashboard', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '6_lesion_comparison_dashboard.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '6_lesion_comparison_dashboard.png'}")
plt.close()

# ============================================================================
# 7. Comprehensive Multi-Lesion Evaluation Report
# ============================================================================
print("\n[7/7] Generating Comprehensive Evaluation Report...")

fig = plt.figure(figsize=(24, 16))
gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4)

# Section 1: Image + Segmentation (top row)
ax_img = fig.add_subplot(gs[0, :2])
ax_img.imshow(base_image)
ax_img.set_title('A. Original Medical Image', fontsize=14, fontweight='bold', loc='left')
ax_img.axis('off')

ax_seg = fig.add_subplot(gs[0, 2:])
ax_seg.imshow(overlay)
ax_seg.set_title('B. Multi-Lesion Segmentation Overlay', fontsize=14, fontweight='bold', loc='left')
ax_seg.axis('off')

# Add legend
legend_elements = [
    mpatches.Patch(color=tuple(c/255 for c in LESION_COLORS[i]), 
                  label=f"Lesion {i+1}: {lesions_info[i]['type']} ({lesions_info[i]['severity']})")
    for i in range(len(lesions_info))
]
ax_seg.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9)

# Section 2: Per-lesion metrics (second row)
# Segmentation quality metrics
ax_seg_metrics = fig.add_subplot(gs[1, 0])
ax_seg_metrics.axis('off')

seg_metrics = {
    'Lesion 1': {'Dice': 0.89, 'IoU': 0.82, 'HD95': 3.2},
    'Lesion 2': {'Dice': 0.84, 'IoU': 0.76, 'HD95': 4.5},
    'Lesion 3': {'Dice': 0.91, 'IoU': 0.85, 'HD95': 2.8},
}

metrics_text = "Segmentation Metrics\n" + "="*25 + "\n\n"
for lesion_name, metrics in seg_metrics.items():
    metrics_text += f"{lesion_name}:\n"
    metrics_text += f"  Dice: {metrics['Dice']:.2f}\n"
    metrics_text += f"  IoU:  {metrics['IoU']:.2f}\n"
    metrics_text += f"  HD95: {metrics['HD95']:.1f}mm\n\n"

ax_seg_metrics.text(0.5, 0.5, metrics_text, transform=ax_seg_metrics.transAxes,
                   fontsize=10, ha='center', va='center', family='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
ax_seg_metrics.set_title('C. Segmentation Quality', fontsize=13, fontweight='bold', loc='left')

# Detection metrics
ax_det_metrics = fig.add_subplot(gs[1, 1])

detection_data = {
    'Precision': 0.94,
    'Recall': 0.91,
    'F1-Score': 0.925,
    'mAP@0.5': 0.89,
}

bars = ax_det_metrics.bar(detection_data.keys(), detection_data.values(), 
                         alpha=0.7, edgecolor='black', linewidth=2)

# Color bars
colors_bars = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for bar, color in zip(bars, colors_bars):
    bar.set_color(color)

ax_det_metrics.set_ylabel('Score', fontsize=11, fontweight='bold')
ax_det_metrics.set_title('D. Detection Performance', fontsize=13, fontweight='bold', loc='left')
ax_det_metrics.set_ylim(0, 1)
ax_det_metrics.grid(axis='y', alpha=0.3)
ax_det_metrics.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
ax_det_metrics.legend(fontsize=9)

# Per-lesion classification accuracy
ax_cls = fig.add_subplot(gs[1, 2])

cls_acc = [0.92, 0.85, 0.78]
lesion_labels = [f"L{i+1}" for i in range(3)]

bars_cls = ax_cls.bar(lesion_labels, cls_acc, alpha=0.7, edgecolor='black', linewidth=2)
for i, bar in enumerate(bars_cls):
    bar.set_color(tuple(c/255 for c in LESION_COLORS[i]))
    ax_cls.text(i, cls_acc[i] + 0.02, f'{cls_acc[i]:.0%}', 
               ha='center', fontsize=11, fontweight='bold')

ax_cls.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax_cls.set_title('E. Per-Lesion Classification', fontsize=13, fontweight='bold', loc='left')
ax_cls.set_ylim(0, 1)
ax_cls.grid(axis='y', alpha=0.3)

# Attention coverage
ax_attn_cov = fig.add_subplot(gs[1, 3])

coverage_data = {
    'L1': {'inside': 0.87, 'outside': 0.13},
    'L2': {'inside': 0.82, 'outside': 0.18},
    'L3': {'inside': 0.91, 'outside': 0.09},
}

inside_vals = [v['inside'] for v in coverage_data.values()]
outside_vals = [v['outside'] for v in coverage_data.values()]

x_pos = np.arange(len(coverage_data))
bars1 = ax_attn_cov.bar(x_pos, inside_vals, 0.6, label='Inside Lesion', 
                       alpha=0.8, color='#2ecc71')
bars2 = ax_attn_cov.bar(x_pos, outside_vals, 0.6, bottom=inside_vals, 
                       label='Outside Lesion', alpha=0.8, color='#e74c3c')

ax_attn_cov.set_ylabel('Attention Proportion', fontsize=11, fontweight='bold')
ax_attn_cov.set_title('F. Attention Localization', fontsize=13, fontweight='bold', loc='left')
ax_attn_cov.set_xticks(x_pos)
ax_attn_cov.set_xticklabels(coverage_data.keys())
ax_attn_cov.legend(fontsize=10)
ax_attn_cov.set_ylim(0, 1)
ax_attn_cov.grid(axis='y', alpha=0.3)

# Section 3: Attention heatmaps for each lesion
for i, lesion in enumerate(lesions_info):
    ax = fig.add_subplot(gs[2, i])
    
    lesion_attn = create_attention_for_lesion(512, lesion['center'], 1.2)
    heat = cv2.applyColorMap((lesion_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over = cv2.addWeighted(base_image, 0.6, heat, 0.4, 0)
    
    ax.imshow(over)
    ax.set_title(f"Lesion {i+1} Attention Heatmap", fontsize=11, fontweight='bold')
    ax.axis('off')

# Global diagnosis summary
ax_diagnosis = fig.add_subplot(gs[2, 3])
ax_diagnosis.axis('off')

diag_text = "🏥 Final Diagnosis\n" + "="*30 + "\n\n"
diag_text += "Primary:\n"
diag_text += "  Multiple pulmonary lesions\n\n"
diag_text += "Findings:\n"
diag_text += "  • L1: Suspected malignancy\n"
diag_text += "      (high risk)\n"
diag_text += "  • L2: Indeterminate mass\n"
diag_text += "      (moderate risk)\n"
diag_text += "  • L3: Likely inflammatory\n"
diag_text += "      (low risk)\n\n"
diag_text += "Confidence: 89%\n"
diag_text += "Urgency: Moderate"

ax_diagnosis.text(0.5, 0.5, diag_text, transform=ax_diagnosis.transAxes,
                 fontsize=10, ha='center', va='center', family='monospace',
                 bbox=dict(boxstyle='round,pad=1', facecolor='#ffe6e6', alpha=0.9))
ax_diagnosis.set_title('G. Global Diagnosis', fontsize=13, fontweight='bold', loc='left')

# Section 4: Reasoning flow diagram
ax_flow = fig.add_subplot(gs[3, :])
ax_flow.axis('off')
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 3)

# Draw reasoning flow
flow_steps = [
    (0.5, 1.5, 'Multi-Image\nInput'),
    (1.8, 1.5, 'Fusion'),
    (3.0, 2.2, 'Segmentation'),
    (3.0, 1.5, 'Detection'),
    (3.0, 0.8, 'Feature Extraction'),
    (4.5, 2.2, 'Per-Lesion\nAnalysis'),
    (4.5, 1.5, 'Global\nReasoning'),
    (4.5, 0.8, 'Attention\nMaps'),
    (6.5, 1.5, 'Multi-Level\nDiagnosis'),
    (8.5, 1.5, 'Final\nReport'),
]

for i, (x, y, text) in enumerate(flow_steps):
    # Draw box
    rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightblue', edgecolor='black',
                                  linewidth=2, alpha=0.7)
    ax_flow.add_patch(rect)
    ax_flow.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw arrow to next
    if i < len(flow_steps) - 1:
        next_x, next_y, _ = flow_steps[i+1]
        if next_x > x:  # Horizontal arrow
            ax_flow.annotate('', xy=(next_x-0.4, next_y), xytext=(x+0.4, y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax_flow.set_title('H. Multi-Lesion Processing Pipeline', 
                 fontsize=13, fontweight='bold', loc='left', pad=20)

plt.suptitle('Comprehensive Multi-Lesion Evaluation Report', 
            fontsize=18, fontweight='bold')
plt.savefig(output_dir / '7_comprehensive_report.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '7_comprehensive_report.png'}")
plt.close()

print("\n" + "=" * 80)
print("All Multi-Lesion visualizations generated successfully!")
print(f"Output directory: {output_dir.absolute()}")
print("=" * 80)
print("\nGenerated files:")
print("  1. 1_multi_lesion_segmentation.png  - Multi-lesion detection & segmentation")
print("  2. 2_multi_image_fusion.png         - MRI multi-sequence fusion")
print("  3. 3_per_lesion_attention.png       - Independent attention per lesion")
print("  4. 4_multi_attention_levels.png     - 3-level attention hierarchy")
print("  5. 5_lesion_reasoning_chains.png    - Per-lesion reasoning steps")
print("  6. 6_lesion_comparison_dashboard.png - Comparative analysis")
print("  7. 7_comprehensive_report.png       - Full evaluation report")
print("\n" + "=" * 80)


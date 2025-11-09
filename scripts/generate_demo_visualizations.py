#!/usr/bin/env python3
"""
Generate Demo Visualizations

Creates demonstration visualizations for the Medical Multimodal CoT framework.
Includes:
1. Attention heatmaps
2. Chain-of-thought visualization
3. Reliability diagram (confidence calibration)
4. Attention localization comparison
5. Comprehensive evaluation dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path("demo_visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Generating Demo Visualizations for Medical Multimodal CoT")
print("=" * 70)

# ============================================================================
# 1. Attention Heatmap Visualization
# ============================================================================
print("\n[1/6] Generating Attention Heatmap...")

# Create synthetic medical image (simulating chest X-ray)
def create_synthetic_medical_image(size=512):
    """Create a synthetic medical image"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background
    img[:, :] = [20, 20, 30]
    
    # Simulated lung regions
    cv2.ellipse(img, (200, 300), (120, 150), 0, 0, 360, (40, 40, 50), -1)
    cv2.ellipse(img, (312, 300), (120, 150), 0, 0, 360, (40, 40, 50), -1)
    
    # Simulated abnormality (brighter region)
    cv2.circle(img, (280, 250), 40, (80, 80, 90), -1)
    cv2.circle(img, (280, 250), 30, (100, 100, 110), -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

# Generate synthetic attention map
def create_synthetic_attention(size=512, center=(280, 250)):
    """Create attention map focused on specific region"""
    y, x = np.ogrid[:size, :size]
    cx, cy = center
    
    # Gaussian-like attention
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    attention = np.exp(-dist**2 / (2 * 60**2))
    
    # Add some secondary attention regions
    attention += 0.3 * np.exp(-((x - 200)**2 + (y - 300)**2) / (2 * 80**2))
    attention += 0.2 * np.exp(-((x - 312)**2 + (y - 300)**2) / (2 * 80**2))
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention

# Create visualizations
medical_image = create_synthetic_medical_image()
attention_map = create_synthetic_attention()

# Apply colormap
heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Create overlay
overlay = cv2.addWeighted(medical_image, 0.6, heatmap, 0.4, 0)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(medical_image)
axes[0].set_title('Original Medical Image\n(Simulated Chest X-ray)', fontsize=14, fontweight='bold')
axes[0].axis('off')

im1 = axes[1].imshow(attention_map, cmap='jet')
axes[1].set_title('Attention Heatmap\n(Model Focus Areas)', fontsize=14, fontweight='bold')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Attention Weight')

axes[2].imshow(overlay)
axes[2].set_title('Attention Overlay\n(Combined View)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.suptitle('Attention Visualization: Where the Model Looks', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '1_attention_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '1_attention_heatmap.png'}")
plt.close()

# ============================================================================
# 2. Chain-of-Thought Visualization
# ============================================================================
print("\n[2/6] Generating Chain-of-Thought Visualization...")

reasoning_steps = [
    {'action': 'Examine overall image', 'observation': 'Bilateral lung fields visible', 'bbox': [50, 100, 450, 450], 'attention': 0.65},
    {'action': 'Focus on left lung', 'observation': 'Normal appearance', 'bbox': [100, 150, 250, 400], 'attention': 0.42},
    {'action': 'Focus on right lung', 'observation': 'Increased opacity noted', 'bbox': [260, 150, 410, 400], 'attention': 0.89},
    {'action': 'Examine abnormal region', 'observation': 'Consolidation pattern present', 'bbox': [240, 200, 320, 300], 'attention': 0.95},
    {'action': 'Correlate with symptoms', 'observation': 'Consistent with infection', 'bbox': [240, 200, 320, 300], 'attention': 0.87},
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, step in enumerate(reasoning_steps):
    img_copy = medical_image.copy()
    
    # Draw bounding box
    bbox = step['bbox']
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    axes[i].imshow(img_copy)
    axes[i].set_title(
        f"Step {i+1}: {step['action']}\nAttention Score: {step['attention']:.2f}",
        fontsize=12,
        fontweight='bold'
    )
    axes[i].axis('off')
    
    # Add observation text
    axes[i].text(
        0.5, -0.08,
        step['observation'],
        transform=axes[i].transAxes,
        ha='center',
        fontsize=10,
        style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
    )

# Hide last subplot
axes[5].axis('off')

plt.suptitle('Chain-of-Thought Reasoning Process', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '2_chain_of_thought.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '2_chain_of_thought.png'}")
plt.close()

# ============================================================================
# 3. Reliability Diagram (Confidence Calibration)
# ============================================================================
print("\n[3/6] Generating Reliability Diagram...")

# Generate synthetic calibration data
np.random.seed(42)
n_samples = 1000

# Simulate model predictions
# Well-calibrated model should have accuracy ≈ confidence
confidences = np.random.beta(3, 2, n_samples)  # Bias towards higher confidence
predictions = np.random.random(n_samples) < confidences * 0.9  # Slightly overconfident

# Bin the data
n_bins = 10
bin_boundaries = np.linspace(0, 1, n_bins + 1)
bin_confidences = []
bin_accuracies = []
bin_counts = []

for i in range(n_bins):
    in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
    if in_bin.sum() > 0:
        bin_confidences.append(confidences[in_bin].mean())
        bin_accuracies.append(predictions[in_bin].mean())
        bin_counts.append(in_bin.sum())
    else:
        bin_confidences.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
        bin_accuracies.append(0)
        bin_counts.append(0)

# Calculate ECE
ece = sum([abs(acc - conf) * count / n_samples 
           for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)])

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Reliability diagram
ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
ax1.plot(bin_confidences, bin_accuracies, 'o-', linewidth=3, 
         markersize=10, label='Model Performance', color='#e74c3c')

# Fill gap area
ax1.fill_between([0, 1], [0, 1], alpha=0.1, color='green', label='Well-Calibrated Region')

for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
    if count > 0:
        ax1.plot([conf, conf], [conf, acc], 'r:', linewidth=1, alpha=0.5)
        ax1.scatter([conf], [acc], s=count/5, alpha=0.3, color='red')

ax1.set_xlabel('Predicted Confidence', fontsize=14, fontweight='bold')
ax1.set_ylabel('Actual Accuracy', fontsize=14, fontweight='bold')
ax1.set_title(f'Reliability Diagram\nECE: {ece:.3f}', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Histogram of confidence distribution
ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
ax2.set_xlabel('Confidence', fontsize=14, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Confidence Calibration Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '3_reliability_diagram.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '3_reliability_diagram.png'}")
plt.close()

# ============================================================================
# 4. Attention Localization Comparison
# ============================================================================
print("\n[4/6] Generating Attention Localization Comparison...")

# Create ground truth mask
gt_mask = np.zeros((512, 512))
cv2.circle(gt_mask, (280, 250), 40, 1, -1)

# Calculate overlap
overlap = (attention_map * gt_mask).sum() / gt_mask.sum()
iou = (((attention_map > 0.5).astype(int) * gt_mask).sum() / 
       ((attention_map > 0.5).astype(int) + gt_mask > 0).sum())

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Masks
axes[0, 0].imshow(medical_image)
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(gt_mask, cmap='Reds', vmin=0, vmax=1)
axes[0, 1].set_title('Ground Truth\n(Expert Annotation)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[0, 2].imshow(attention_map, cmap='Reds', vmin=0, vmax=1)
axes[0, 2].set_title('Model Attention\n(AI Prediction)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# Row 2: Overlays and comparison
overlay_gt = medical_image.copy()
overlay_gt[:, :, 0] = np.clip(overlay_gt[:, :, 0] + (gt_mask * 100).astype(np.uint8), 0, 255)

overlay_pred = medical_image.copy()
overlay_pred[:, :, 0] = np.clip(overlay_pred[:, :, 0] + (attention_map * 100).astype(np.uint8), 0, 255)

axes[1, 0].imshow(overlay_gt)
axes[1, 0].set_title('GT Overlay', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay_pred)
axes[1, 1].set_title('Attention Overlay', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Comparison metrics
comparison = np.zeros((512, 512, 3), dtype=np.uint8)
comparison[:, :] = medical_image

# True Positive (both agree) - Green
tp_mask = ((attention_map > 0.5) & (gt_mask > 0))
comparison[tp_mask, 1] = np.clip(comparison[tp_mask, 1] + 100, 0, 255)

# False Positive (model only) - Red
fp_mask = ((attention_map > 0.5) & (gt_mask == 0))
comparison[fp_mask, 0] = np.clip(comparison[fp_mask, 0] + 100, 0, 255)

# False Negative (GT only) - Blue
fn_mask = ((attention_map <= 0.5) & (gt_mask > 0))
comparison[fn_mask, 2] = np.clip(comparison[fn_mask, 2] + 100, 0, 255)

axes[1, 2].imshow(comparison)
axes[1, 2].set_title(f'Comparison\nOverlap: {overlap:.3f} | IoU: {iou:.3f}', 
                     fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.5, label='True Positive (Correct)'),
    Patch(facecolor='red', alpha=0.5, label='False Positive (Over-attention)'),
    Patch(facecolor='blue', alpha=0.5, label='False Negative (Missed)'),
]
axes[1, 2].legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.suptitle('Attention Localization Accuracy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '4_attention_localization.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '4_attention_localization.png'}")
plt.close()

# ============================================================================
# 5. Deletion/Insertion Curves
# ============================================================================
print("\n[5/6] Generating Deletion/Insertion Curves...")

# Simulate deletion and insertion
steps = np.linspace(0, 1, 20)
deletion_scores = 1.0 - steps ** 1.5 + np.random.normal(0, 0.05, len(steps))
insertion_scores = steps ** 0.8 + np.random.normal(0, 0.05, len(steps))

deletion_scores = np.clip(deletion_scores, 0, 1)
insertion_scores = np.clip(insertion_scores, 0, 1)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.plot(steps * 100, deletion_scores, 'o-', linewidth=3, markersize=8,
        label='Deletion Curve', color='#e74c3c')
ax.plot(steps * 100, insertion_scores, 's-', linewidth=3, markersize=8,
        label='Insertion Curve', color='#2ecc71')

# Calculate AUC
deletion_auc = np.trapz(deletion_scores, steps)
insertion_auc = np.trapz(insertion_scores, steps)

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
ax.fill_between(steps * 100, deletion_scores, alpha=0.2, color='red')
ax.fill_between(steps * 100, insertion_scores, alpha=0.2, color='green')

ax.set_xlabel('Percentage of Image Modified (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Confidence', fontsize=14, fontweight='bold')
ax.set_title(f'Attention Importance Analysis\nDeletion AUC: {deletion_auc:.3f} | Insertion AUC: {insertion_auc:.3f}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='center right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.1)

# Add annotations
ax.annotate('Lower is better\n(attention is important)',
            xy=(80, deletion_scores[-5]), xytext=(60, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

ax.annotate('Higher is better\n(attention is sufficient)',
            xy=(80, insertion_scores[-5]), xytext=(60, 0.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '5_deletion_insertion.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '5_deletion_insertion.png'}")
plt.close()

# ============================================================================
# 6. Comprehensive Evaluation Dashboard
# ============================================================================
print("\n[6/6] Generating Comprehensive Dashboard...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Metrics summary (top row)
ax_metrics = fig.add_subplot(gs[0, :])
ax_metrics.axis('off')

metrics_data = {
    'Classification': ['Accuracy: 89.2%', 'F1-Score: 0.91', 'AUC-ROC: 0.94'],
    'Confidence': ['ECE: 0.032', 'Brier Score: 0.041', 'Calibration: Good'],
    'Attention': ['Overlap: 0.87', 'Point Acc: 92%', 'Del-AUC: 0.23'],
    'Reasoning': ['Consistency: 0.78', 'Coherence: 0.85', 'Expert Agree: 81%']
}

x_pos = [0.12, 0.37, 0.62, 0.87]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (title, metrics) in enumerate(metrics_data.items()):
    # Box
    rect = Rectangle((x_pos[i] - 0.11, 0.3), 0.22, 0.6, 
                      facecolor=colors[i], alpha=0.2, transform=ax_metrics.transAxes)
    ax_metrics.add_patch(rect)
    
    # Title
    ax_metrics.text(x_pos[i], 0.85, title, transform=ax_metrics.transAxes,
                   fontsize=14, fontweight='bold', ha='center')
    
    # Metrics
    for j, metric in enumerate(metrics):
        ax_metrics.text(x_pos[i], 0.70 - j*0.15, metric,
                       transform=ax_metrics.transAxes,
                       fontsize=11, ha='center')

ax_metrics.set_title('Medical AI Evaluation Dashboard', fontsize=18, fontweight='bold', pad=20)

# Confusion Matrix
ax_cm = fig.add_subplot(gs[1, 0])
cm = np.array([[85, 5, 3], [4, 78, 6], [2, 5, 82]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
            xticklabels=['Normal', 'Pneumonia', 'Other'],
            yticklabels=['Normal', 'Pneumonia', 'Other'])
ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax_cm.set_ylabel('True Label', fontweight='bold')
ax_cm.set_xlabel('Predicted Label', fontweight='bold')

# Reliability mini
ax_rel = fig.add_subplot(gs[1, 1])
ax_rel.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
ax_rel.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=6, color='#e74c3c')
ax_rel.set_xlabel('Confidence', fontweight='bold')
ax_rel.set_ylabel('Accuracy', fontweight='bold')
ax_rel.set_title('Calibration Curve', fontsize=12, fontweight='bold')
ax_rel.grid(True, alpha=0.3)

# ROC Curve
ax_roc = fig.add_subplot(gs[1, 2])
fpr = np.linspace(0, 1, 100)
tpr = 1 - (1 - fpr) ** 1.3 + np.random.normal(0, 0.02, 100)
tpr = np.clip(tpr, 0, 1)
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random')
ax_roc.plot(fpr, tpr, linewidth=3, color='#2ecc71', label='Model (AUC=0.94)')
ax_roc.fill_between(fpr, tpr, alpha=0.2, color='green')
ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
ax_roc.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax_roc.legend(loc='lower right')
ax_roc.grid(True, alpha=0.3)

# Attention scores distribution
ax_att = fig.add_subplot(gs[1, 3])
att_scores = np.random.beta(4, 2, 500)
ax_att.hist(att_scores, bins=30, edgecolor='black', alpha=0.7, color='#f39c12')
ax_att.axvline(att_scores.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {att_scores.mean():.2f}')
ax_att.set_xlabel('Attention Score', fontweight='bold')
ax_att.set_ylabel('Frequency', fontweight='bold')
ax_att.set_title('Attention Distribution', fontsize=12, fontweight='bold')
ax_att.legend()
ax_att.grid(True, alpha=0.3, axis='y')

# Bottom row: Example attention visualizations
for i in range(4):
    ax_ex = fig.add_subplot(gs[2, i])
    
    # Create small example
    small_img = cv2.resize(medical_image, (128, 128))
    small_att = create_synthetic_attention(128, center=(70 + i*10, 64 + i*5))
    small_heat = cv2.applyColorMap((small_att * 255).astype(np.uint8), cv2.COLORMAP_JET)
    small_heat = cv2.cvtColor(small_heat, cv2.COLOR_BGR2RGB)
    small_overlay = cv2.addWeighted(small_img, 0.6, small_heat, 0.4, 0)
    
    ax_ex.imshow(small_overlay)
    quality = ['Excellent', 'Good', 'Fair', 'Poor'][i]
    colors_q = ['green', 'blue', 'orange', 'red']
    ax_ex.set_title(f'Case {i+1}: {quality}', fontsize=10, 
                   fontweight='bold', color=colors_q[i])
    ax_ex.axis('off')

plt.savefig(output_dir / '6_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '6_evaluation_dashboard.png'}")
plt.close()

print("\n" + "=" * 70)
print("All visualizations generated successfully!")
print(f"Output directory: {output_dir.absolute()}")
print("=" * 70)


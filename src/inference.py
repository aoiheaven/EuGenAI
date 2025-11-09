"""
Inference and Visualization Module

Provides tools for:
- Model inference on new medical images
- Attention visualization
- Chain-of-thought explanation generation
- Interactive result exploration
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torchvision import transforms

from model import MedicalMultimodalCoT
from dataset import MedicalChainOfThoughtDataset
from utils import TextProcessor, DiagnosisLabelEncoder


class MedicalCoTInference:
    """Inference engine for medical multimodal chain-of-thought model"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Optional[Dict] = None,
        label_file: str = 'data/diagnosis_labels.json',
    ):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            config: Model configuration (optional, will load from checkpoint if not provided)
            label_file: Path to diagnosis label mappings
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if config is None:
            config = checkpoint.get('config', {}).get('model', {})
        
        # Initialize model
        self.model = MedicalMultimodalCoT(**config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Initialize text processor
        text_encoder_name = config.get('text_encoder_name', 'bert-base-uncased')
        self.text_processor = TextProcessor(
            tokenizer_name=text_encoder_name,
            max_length=512
        )
        
        # Load diagnosis label encoder
        self.label_encoder = DiagnosisLabelEncoder(label_file)
        
        # Image transforms
        img_size = config.get('img_size', 512)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        clinical_text: str,
        reasoning_steps: List[Dict],
    ) -> Dict:
        """
        Run inference on a medical case
        
        Args:
            image_path: Path to medical image
            clinical_text: Clinical text (history, exam, labs)
            reasoning_steps: List of reasoning steps with regions
            
        Returns:
            Dictionary containing predictions and attention maps
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Parse clinical text (assuming format: "history | exam | labs")
        # If not in this format, treat entire text as history
        if '|' in clinical_text:
            parts = clinical_text.split('|')
            history = parts[0].strip() if len(parts) > 0 else ''
            exam = parts[1].strip() if len(parts) > 1 else ''
            labs = parts[2].strip() if len(parts) > 2 else ''
        else:
            history = clinical_text
            exam = ''
            labs = ''
        
        # Encode clinical text
        encoded_text = self.text_processor.encode_clinical_text(history, exam, labs)
        text_input_ids = encoded_text['input_ids'].unsqueeze(0).to(self.device)
        text_attention_mask = encoded_text['attention_mask'].unsqueeze(0).to(self.device)
        
        # Encode reasoning steps
        if reasoning_steps:
            encoded_steps = self.text_processor.encode_reasoning_steps(
                reasoning_steps,
                max_steps=10
            )
            cot_step_input_ids = encoded_steps['input_ids'].unsqueeze(0).to(self.device)
            cot_step_attention_mask = encoded_steps['attention_mask'].unsqueeze(0).to(self.device)
            num_steps = encoded_steps['num_steps']
        else:
            # No reasoning steps provided - create empty placeholders
            num_steps = 0
            max_steps = 10
            seq_len = 512
            cot_step_input_ids = torch.zeros(1, max_steps, seq_len, dtype=torch.long, device=self.device)
            cot_step_attention_mask = torch.zeros(1, max_steps, seq_len, dtype=torch.long, device=self.device)
        
        # Extract bounding boxes
        max_steps = 10
        bboxes = []
        for step in reasoning_steps[:max_steps]:
            bbox = step.get('region_of_interest', {}).get('bbox', [0, 0, 0, 0])
            # Ensure bbox is always a list of 4 numbers
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]
            bboxes.append(bbox)
        
        # Pad to max_steps
        while len(bboxes) < max_steps:
            bboxes.append([0, 0, 0, 0])
        
        cot_step_regions = torch.tensor([bboxes], dtype=torch.float32, device=self.device)
        
        cot_num_steps = torch.tensor([num_steps], device=self.device)
        
        # Forward pass
        outputs = self.model(
            images=image_tensor,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            cot_step_input_ids=cot_step_input_ids,
            cot_step_attention_mask=cot_step_attention_mask,
            cot_step_regions=cot_step_regions,
            cot_num_steps=cot_num_steps,
        )
        
        # Process outputs
        diagnosis_probs = torch.softmax(outputs['diagnosis_logits'], dim=-1)
        top_k_probs, top_k_indices = torch.topk(diagnosis_probs, k=min(5, diagnosis_probs.size(-1)), dim=-1)
        
        # Decode diagnosis labels
        top_diagnoses = []
        for idx, prob in zip(top_k_indices[0], top_k_probs[0]):
            diagnosis_text = self.label_encoder.decode(idx.item())
            top_diagnoses.append({
                'diagnosis': diagnosis_text,
                'probability': prob.item()
            })
        
        results = {
            'top_diagnoses': top_diagnoses,
            'confidence': outputs['confidence'].item(),
            'step_attentions': outputs['step_attentions'].cpu().numpy(),
            'cross_modal_attention': outputs['cross_modal_attention'].cpu().numpy(),
        }
        
        return results
    
    def visualize_attention(
        self,
        image_path: str,
        attention_map: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Attention Heatmap",
    ):
        """
        Visualize attention heatmap overlaid on image
        
        Args:
            image_path: Path to original image
            attention_map: Attention weights [H, W] or [num_patches]
            save_path: Path to save visualization
            title: Plot title
        """
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize attention map to match image
        if len(attention_map.shape) == 1:
            # Reshape patch attention to 2D
            side_len = int(np.sqrt(len(attention_map)))
            attention_map = attention_map.reshape(side_len, side_len)
        
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Normalize attention
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attention_map, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_chain_of_thought(
        self,
        image_path: str,
        reasoning_steps: List[Dict],
        step_attentions: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Visualize chain-of-thought reasoning process
        
        Args:
            image_path: Path to medical image
            reasoning_steps: List of reasoning steps
            step_attentions: Attention weights for each step
            save_path: Path to save visualization
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        num_steps = len(reasoning_steps)
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if num_steps == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (step, attention) in enumerate(zip(reasoning_steps, step_attentions)):
            img_copy = image.copy()
            
            # Draw bounding box if exists
            bbox = step.get('bbox', None)
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            axes[i].imshow(img_copy)
            axes[i].set_title(f"Step {i+1}: {step.get('action', '')}\nAttention: {attention:.3f}", 
                             fontsize=10)
            axes[i].axis('off')
            
            # Add observation text
            obs = step.get('observation', '')
            if len(obs) > 60:
                obs = obs[:57] + '...'
            axes[i].text(0.5, -0.1, obs, transform=axes[i].transAxes,
                        ha='center', va='top', fontsize=8, wrap=True)
        
        # Hide unused subplots
        for i in range(num_steps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Chain-of-Thought Reasoning Process', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CoT visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(
        self,
        image_path: str,
        clinical_text: str,
        reasoning_steps: List[Dict],
        output_dir: str = 'outputs',
    ):
        """
        Generate comprehensive diagnostic report with visualizations
        
        Args:
            image_path: Path to medical image
            clinical_text: Clinical information
            reasoning_steps: Chain-of-thought steps
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        results = self.predict(image_path, clinical_text, reasoning_steps)
        
        # Create visualizations
        print("Generating visualizations...")
        
        # 1. Attention heatmap
        if 'cross_modal_attention' in results:
            attention = results['cross_modal_attention'][0].mean(axis=0)  # Average over heads
            self.visualize_attention(
                image_path,
                attention,
                save_path=str(output_path / 'attention_heatmap.png'),
            )
        
        # 2. Chain-of-thought
        if 'step_attentions' in results:
            self.visualize_chain_of_thought(
                image_path,
                reasoning_steps,
                results['step_attentions'][0],
                save_path=str(output_path / 'chain_of_thought.png'),
            )
        
        # 3. Save report JSON
        report = {
            'image': image_path,
            'clinical_text': clinical_text,
            'predictions': results['top_diagnoses'],
            'confidence': results['confidence'],
            'reasoning_steps': reasoning_steps,
        }
        
        with open(output_path / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport generated in {output_dir}/")
        print(f"Top diagnosis: Index {results['top_diagnoses'][0]['index']} "
              f"(Prob: {results['top_diagnoses'][0]['probability']:.3f})")
        print(f"Confidence: {results['confidence']:.3f}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Medical CoT Model Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to medical image')
    parser.add_argument('--text', type=str, default='', help='Clinical text')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = MedicalCoTInference(args.checkpoint, device=args.device)
    
    # Example reasoning steps (in practice, these would come from user input or another model)
    example_steps = [
        {
            'action': 'Examine overall image',
            'observation': 'Chest CT scan showing cardiac region',
            'bbox': [100, 100, 400, 400],
        },
        {
            'action': 'Focus on abnormal region',
            'observation': 'Increased opacity in left lung field',
            'bbox': [150, 150, 350, 350],
        },
    ]
    
    # Generate report
    engine.generate_report(
        image_path=args.image,
        clinical_text=args.text,
        reasoning_steps=example_steps,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Generate synthetic fundus images for testing

Creates simple circular images that mimic fundus photos for pipeline testing.
NOT for production use - only for verifying code works.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import argparse
from pathlib import Path
import random


def generate_synthetic_fundus(size=(224, 224), seed=None):
    """
    Generate a synthetic fundus image
    
    Args:
        size: Image size (width, height)
        seed: Random seed for reproducibility
    
    Returns:
        PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create base image (dark red/orange background)
    img = Image.new('RGB', size, color=(120, 50, 30))
    draw = ImageDraw.Draw(img)
    
    # Draw fundus-like circle (bright center, dark edges)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 2 - 10
    
    # Main fundus circle
    for r in range(radius, 0, -5):
        brightness = int(255 * (r / radius))
        color = (
            min(255, 150 + brightness // 3),
            min(255, 80 + brightness // 2),
            min(255, 40 + brightness // 4)
        )
        draw.ellipse(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
            fill=color
        )
    
    # Add optic disc (bright spot)
    disc_x = center[0] - radius // 3
    disc_y = center[1]
    disc_radius = radius // 6
    draw.ellipse(
        [disc_x - disc_radius, disc_y - disc_radius, 
         disc_x + disc_radius, disc_y + disc_radius],
        fill=(255, 220, 180)
    )
    
    # Add some vessel-like lines
    for _ in range(5):
        start_x = disc_x + random.randint(-10, 10)
        start_y = disc_y + random.randint(-10, 10)
        end_x = center[0] + random.randint(-radius, radius)
        end_y = center[1] + random.randint(-radius, radius)
        draw.line(
            [(start_x, start_y), (end_x, end_y)],
            fill=(180, 60, 40),
            width=2
        )
    
    # Add random spots (simulate lesions for different DR grades)
    num_spots = random.randint(0, 15)
    for _ in range(num_spots):
        spot_x = center[0] + random.randint(-radius // 2, radius // 2)
        spot_y = center[1] + random.randint(-radius // 2, radius // 2)
        spot_radius = random.randint(2, 6)
        spot_color = (
            random.randint(100, 200),
            random.randint(30, 80),
            random.randint(20, 50)
        )
        draw.ellipse(
            [spot_x - spot_radius, spot_y - spot_radius,
             spot_x + spot_radius, spot_y + spot_radius],
            fill=spot_color
        )
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Add slight noise
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic fundus images for testing"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=30,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/images/quick_test"),
        help="Output directory"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Image size (will be square)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Generating {args.num_images} synthetic fundus images...")
    print(f"   Output: {args.output_dir}")
    print(f"   Size: {args.size}x{args.size}")
    print()
    
    # Generate images with different names matching dataset
    splits = ["train", "val", "test"]
    counts = {
        "train": 20,
        "val": 5,
        "test": 5
    }
    
    total_generated = 0
    for split in splits:
        for i in range(counts.get(split, 0)):
            if total_generated >= args.num_images:
                break
                
            filename = f"{split}_{i:04d}.jpg"
            filepath = args.output_dir / filename
            
            # Generate image with seed for reproducibility
            img = generate_synthetic_fundus(
                size=(args.size, args.size),
                seed=total_generated
            )
            
            # Save image
            img.save(filepath, quality=95)
            
            total_generated += 1
            
            if (total_generated) % 10 == 0:
                print(f"   Generated {total_generated}/{args.num_images}...")
    
    print(f"\n✅ Successfully generated {total_generated} images!")
    print(f"   Location: {args.output_dir}")
    print()
    print("⚠️  Note: These are synthetic images for TESTING ONLY")
    print("   For real training, use actual fundus images from:")
    print("   - Kaggle: diabetic-retinopathy-detection dataset")
    print("   - APTOS 2019 Blindness Detection")
    print("   - Messidor / Messidor-2")
    print()


if __name__ == "__main__":
    main()


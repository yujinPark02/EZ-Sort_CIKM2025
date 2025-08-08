
# examples/medical_example.py
"""
Example: Medical Image Quality Assessment with EZ-Sort
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig


def create_medical_dataset(n_images=40, output_dir="sample_medical"):
    """Create synthetic medical image dataset"""
    
    print(f"Creating medical image dataset with {n_images} images...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Quality levels: 0=poor, 1=fair, 2=good, 3=excellent
    qualities = np.random.choice([0, 1, 2, 3], n_images, p=[0.2, 0.3, 0.3, 0.2])
    
    image_data = []
    
    for i, quality in enumerate(qualities):
        img_name = f"medical_{i:03d}.jpg"
        
        # Create base medical image (simulated X-ray)
        img = Image.new('L', (256, 256), color=40)  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Add "anatomical" structures
        # Ribcage
        for rib in range(8):
            y = 50 + rib * 20
            draw.arc([50, y, 200, y + 15], 0, 180, fill=180, width=2)
        
        # Spine
        draw.line([(128, 30), (128, 220)], fill=200, width=4)
        
        # Heart shadow
        draw.ellipse([90, 80, 140, 130], fill=120)
        
        # Add quality-based degradation
        if quality == 0:  # Poor quality
            # Heavy noise
            pixels = np.array(img)
            noise = np.random.normal(0, 30, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255)
            img = Image.fromarray(pixels.astype(np.uint8))
            
            # Heavy blur
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
            
        elif quality == 1:  # Fair quality
            # Moderate noise
            pixels = np.array(img)
            noise = np.random.normal(0, 15, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255)
            img = Image.fromarray(pixels.astype(np.uint8))
            
            # Slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            
        elif quality == 2:  # Good quality
            # Light noise
            pixels = np.array(img)
            noise = np.random.normal(0, 8, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255)
            img = Image.fromarray(pixels.astype(np.uint8))
            
        # quality == 3 (Excellent) gets no degradation
        
        # Convert back to RGB for CLIP
        img = img.convert('RGB')
        
        # Save image
        img_path = os.path.join(output_dir, "images", img_name)
        img.save(img_path)
        
        image_data.append({
            'image_path': img_name,
            'quality': quality,
            'quality_label': ['poor', 'fair', 'good', 'excellent'][quality]
        })
    
    # Create CSV
    df = pd.DataFrame(image_data)
    csv_path = os.path.join(output_dir, "medical_quality.csv")
    df.to_csv(csv_path, index=False)
    
    quality_dist = np.bincount(qualities)
    print(f"✅ Created {n_images} medical images")
    print(f"   Quality distribution: Poor={quality_dist[0]}, Fair={quality_dist[1]}, Good={quality_dist[2]}, Excellent={quality_dist[3]}")
    
    return csv_path, os.path.join(output_dir, "images")


def run_medical_annotation():
    """Run medical image quality annotation"""
    
    print(f"\n🏥 Medical Image Quality Assessment with EZ-Sort")
    print("=" * 50)
    
    # Create dataset
    csv_path, image_dir = create_medical_dataset(40)
    
    # Medical-specific configuration
    config = EZSortConfig(
        domain="medical",
        range_description="poor to excellent quality",
        k_buckets=3,  # Fewer buckets for medical
        theta_0=0.1,  # Lower threshold (more human oversight)
        hierarchical_prompts={
            "level_1": [
                "a high quality medical image with clear anatomical structures",
                "a low quality medical image with poor visibility or artifacts"
            ],
            "level_2": [
                "an excellent quality medical image with perfect clarity and no artifacts",
                "a good quality medical image with minor imperfections but diagnostic quality",
                "a poor quality medical image with significant artifacts or poor contrast",
                "a very poor quality medical image unsuitable for diagnosis"
            ]
        }
    )
    
    # Load and process
    dataset = EZSortDataset(csv_path, image_dir, "image_path", "quality")
    
    try:
        annotator = EZSortAnnotator(dataset, config)
        results = annotator.run_annotation_session(60)
        
        print(f"\n📊 Medical Quality Assessment Results:")
        print(f"   Automation rate: {results['automation_rate']:.1%}")
        
        if 'final_ranking' in results:
            print(f"\n🏆 Top 5 Highest Quality Images:")
            for i, idx in enumerate(results['final_ranking'][:5]):
                quality = dataset.labels[idx]
                img_name = dataset.image_paths[idx]
                quality_label = ['poor', 'fair', 'good', 'excellent'][quality]
                print(f"   {i+1}. {img_name} - Quality: {quality_label}")
        
        return results
        
    except Exception as e:
        print(f"❌ Medical annotation failed: {e}")
        return None


if __name__ == "__main__":
    run_medical_annotation()


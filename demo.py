# demo.py
"""
EZ-Sort Demo Script
Demonstrates the basic usage of EZ-Sort with synthetic data.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig
import argparse


def create_synthetic_face_dataset(n_images: int = 50, output_dir: str = "demo_data"):
    """Create synthetic face dataset for demonstration"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Generate synthetic data
    ages = np.random.randint(1, 70, n_images)
    names = [f"face_{i:03d}.jpg" for i in range(n_images)]
    
    # Create synthetic face images (placeholder rectangles with age text)
    image_data = []
    
    for i, age in enumerate(ages):
        # Create image based on age
        img_size = (224, 224)
        
        # Color based on age group
        if age < 10:
            color = (255, 220, 220)  # Light pink for babies/children
        elif age < 20:
            color = (220, 255, 220)  # Light green for teenagers
        elif age < 40:
            color = (220, 220, 255)  # Light blue for young adults
        elif age < 60:
            color = (255, 255, 220)  # Light yellow for adults
        else:
            color = (240, 240, 240)  # Light gray for elderly
        
        # Create image
        img = Image.new('RGB', img_size, color)
        draw = ImageDraw.Draw(img)
        
        # Draw age text
        try:
            # Try to use a font (may not be available on all systems)
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = f"Age: {age}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
        draw.text(position, text, fill=(0, 0, 0), font=font)
        
        # Add some simple "facial features" (circles)
        # Eyes
        eye_y = img_size[1] // 3
        draw.ellipse([50, eye_y, 70, eye_y + 20], fill=(0, 0, 0))
        draw.ellipse([154, eye_y, 174, eye_y + 20], fill=(0, 0, 0))
        
        # Nose (triangle)
        nose_points = [112, eye_y + 30, 102, eye_y + 50, 122, eye_y + 50]
        draw.polygon(nose_points, fill=(100, 100, 100))
        
        # Mouth
        mouth_y = eye_y + 70
        draw.arc([90, mouth_y, 134, mouth_y + 20], 0, 180, fill=(0, 0, 0), width=3)
        
        # Save image
        img_path = os.path.join(output_dir, "images", names[i])
        img.save(img_path)
        
        image_data.append({
            "image_path": names[i],
            "age": age,
            "age_group": "child" if age < 18 else "adult"
        })
    
    # Create CSV
    df = pd.DataFrame(image_data)
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created synthetic dataset:")
    print(f"   📁 Directory: {output_dir}")
    print(f"   🖼️ Images: {n_images}")
    print(f"   📊 CSV: {csv_path}")
    print(f"   📈 Age range: {ages.min()}-{ages.max()} years")
    
    return csv_path, os.path.join(output_dir, "images")


def run_demo_simulation(csv_path: str, image_dir: str, max_comparisons: int = 50):
    """Run EZ-Sort demo simulation"""
    
    print("\n🎯 Running EZ-Sort Demo Simulation")
    print("=" * 50)
    
    # Load configuration
    config = EZSortConfig(
        domain="face",
        k_buckets=5,
        theta_0=0.15
    )
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = EZSortDataset(csv_path, image_dir, "image_path", "age")
    
    # Initialize EZ-Sort
    print("🧠 Initializing EZ-Sort...")
    annotator = EZSortAnnotator(dataset, config)
    
    # Run simulation
    print(f"🚀 Running annotation simulation ({max_comparisons} comparisons)...")
    results = annotator.run_annotation_session(max_comparisons)
    
    # Print results
    print("\n📊 Results:")
    print(f"   Total comparisons: {results['total_comparisons']}")
    print(f"   Human queries: {results['human_queries']}")
    print(f"   Auto decisions: {results['auto_decisions']}")
    print(f"   Automation rate: {results['automation_rate']:.1%}")
    
    # Show top rankings
    if 'final_ranking' in results:
        print(f"\n🏆 Top 10 Rankings (by age):")
        top_10 = results['final_ranking'][:10]
        for i, idx in enumerate(top_10):
            age = dataset.labels[idx]
            image_name = dataset.image_paths[idx]
            print(f"   {i+1:2d}. {image_name} (Age: {age})")
    
    # Calculate accuracy if ground truth available
    if 'final_ranking' in results:
        from scipy.stats import spearmanr
        
        # Get predicted and true rankings
        predicted_ages = [dataset.labels[idx] for idx in results['final_ranking']]
        true_ranking = np.argsort(dataset.labels)[::-1]  # Descending age order
        pred_ranking = results['final_ranking']
        
        # Calculate Spearman correlation
        correlation, p_value = spearmanr(true_ranking, pred_ranking)
        print(f"\n📈 Ranking Accuracy:")
        print(f"   Spearman correlation: {correlation:.3f}")
        print(f"   P-value: {p_value:.6f}")
    
    return results


def main():
    """Main demo function"""
    
    parser = argparse.ArgumentParser(description="EZ-Sort Demo")
    parser.add_argument("--n-images", type=int, default=30, help="Number of synthetic images to create")
    parser.add_argument("--max-comparisons", type=int, default=50, help="Maximum comparisons in simulation")
    parser.add_argument("--data-dir", type=str, default="demo_data", help="Directory for demo data")
    parser.add_argument("--skip-creation", action="store_true", help="Skip dataset creation (use existing)")
    
    args = parser.parse_args()
    
    print("🎯 EZ-Sort Demo")
    print("=" * 30)
    
    # Create or use existing dataset
    if not args.skip_creation:
        print("📦 Creating synthetic face dataset...")
        csv_path, image_dir = create_synthetic_face_dataset(args.n_images, args.data_dir)
    else:
        csv_path = os.path.join(args.data_dir, "dataset.csv")
        image_dir = os.path.join(args.data_dir, "images")
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            print("   Run without --skip-creation to create demo data first")
            return
    
    # Run demo
    try:
        results = run_demo_simulation(csv_path, image_dir, args.max_comparisons)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Try the web interface: streamlit run ez_sort_web.py")
        print(f"   2. Use your own data: python ez_sort_cli.py --csv your_data.csv --images your_images/")
        print(f"   3. Customize prompts: edit configs/face_age_config.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


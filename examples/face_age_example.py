# examples/face_age_example.py
"""
Example: Face Age Estimation with EZ-Sort
Demonstrates how to use EZ-Sort for face age estimation tasks.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig
from utils import calculate_ranking_metrics, visualize_results, export_results_to_csv


def create_sample_face_dataset(n_faces=50, output_dir="sample_faces"):
    """Create a sample face age dataset"""
    
    print(f"Creating sample face dataset with {n_faces} faces...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Generate realistic age distribution
    ages = []
    for _ in range(n_faces):
        if np.random.random() < 0.3:  # 30% children
            age = np.random.randint(1, 18)
        elif np.random.random() < 0.6:  # 60% adults
            age = np.random.randint(18, 65)
        else:  # 10% elderly
            age = np.random.randint(65, 90)
        ages.append(age)
    
    # Create synthetic face images
    image_data = []
    for i, age in enumerate(ages):
        img_name = f"face_{i:03d}.jpg"
        
        # Create age-representative image
        img_size = (224, 224)
        
        # Base skin color
        base_color = (220, 190, 160)
        img = Image.new('RGB', img_size, base_color)
        draw = ImageDraw.Draw(img)
        
        # Age-based facial features
        if age < 10:
            # Baby/child features
            eye_size = 25
            face_ratio = 0.8
            nose_size = 8
        elif age < 20:
            # Teenager features
            eye_size = 20
            face_ratio = 0.85
            nose_size = 12
        elif age < 50:
            # Adult features
            eye_size = 18
            face_ratio = 0.9
            nose_size = 15
        else:
            # Elderly features
            eye_size = 16
            face_ratio = 0.95
            nose_size = 18
        
        # Draw basic face structure
        face_center = (img_size[0] // 2, img_size[1] // 2)
        face_width = int(img_size[0] * 0.7 * face_ratio)
        face_height = int(img_size[1] * 0.8 * face_ratio)
        
        # Face outline
        face_bbox = [
            face_center[0] - face_width // 2,
            face_center[1] - face_height // 2,
            face_center[0] + face_width // 2,
            face_center[1] + face_height // 2
        ]
        draw.ellipse(face_bbox, fill=base_color, outline=(100, 80, 60), width=3)
        
        # Eyes
        eye_y = face_center[1] - 20
        left_eye = [face_center[0] - 30, eye_y - eye_size//2, 
                   face_center[0] - 30 + eye_size, eye_y + eye_size//2]
        right_eye = [face_center[0] + 30 - eye_size, eye_y - eye_size//2,
                    face_center[0] + 30, eye_y + eye_size//2]
        
        draw.ellipse(left_eye, fill=(50, 30, 20))
        draw.ellipse(right_eye, fill=(50, 30, 20))
        
        # Nose
        nose_points = [
            face_center[0], face_center[1] - 10,
            face_center[0] - nose_size//2, face_center[1] + 10,
            face_center[0] + nose_size//2, face_center[1] + 10
        ]
        draw.polygon(nose_points, fill=(180, 160, 140))
        
        # Mouth
        mouth_y = face_center[1] + 30
        mouth_width = 40 if age > 20 else 30
        draw.arc([face_center[0] - mouth_width//2, mouth_y - 10,
                 face_center[0] + mouth_width//2, mouth_y + 10], 
                 0, 180, fill=(120, 80, 60), width=3)
        
        # Add age-specific details
        if age > 40:
            # Wrinkles
            for j in range(3):
                y_pos = face_center[1] - 40 + j * 15
                draw.line([(face_center[0] - 60, y_pos), (face_center[0] + 60, y_pos)], 
                         fill=(150, 130, 110), width=1)
        
        if age > 60:
            # More pronounced aging
            draw.line([(face_center[0] - 25, face_center[1] - 50), 
                      (face_center[0] + 25, face_center[1] - 50)], 
                     fill=(130, 110, 90), width=2)
        
        # Add age text for reference
        draw.text((10, 10), f"Age: {age}", fill=(255, 255, 255))
        
        # Save image
        img_path = os.path.join(output_dir, "images", img_name)
        img.save(img_path)
        
        image_data.append({
            'image_path': img_name,
            'age': age,
            'age_group': 'child' if age < 18 else 'adult' if age < 65 else 'elderly'
        })
    
    # Create CSV
    df = pd.DataFrame(image_data)
    csv_path = os.path.join(output_dir, "face_ages.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created {n_faces} synthetic faces")
    print(f"   📁 Directory: {output_dir}")
    print(f"   📊 Age range: {min(ages)} - {max(ages)} years")
    print(f"   📈 Mean age: {np.mean(ages):.1f} years")
    
    return csv_path, os.path.join(output_dir, "images")


def run_face_age_annotation(csv_path, image_dir, max_comparisons=100):
    """Run face age annotation with EZ-Sort"""
    
    print(f"\n🎯 Running Face Age Annotation with EZ-Sort")
    print("=" * 50)
    
    # Configure EZ-Sort for face age estimation
    config = EZSortConfig(
        domain="face",
        range_description="0-90 years",
        k_buckets=5,
        theta_0=0.15,
        alpha=0.3,
        beta=0.9
    )
    
    # Load dataset
    print("📂 Loading face dataset...")
    dataset = EZSortDataset(csv_path, image_dir, "image_path", "age")
    
    print(f"   Loaded {dataset.n_items} faces")
    print(f"   Age range: {min(dataset.labels)} - {max(dataset.labels)} years")
    
    # Initialize EZ-Sort
    print("\n🧠 Initializing EZ-Sort (running CLIP hierarchical classification)...")
    try:
        annotator = EZSortAnnotator(dataset, config)
        
        print(f"   ✅ CLIP pre-ordering completed")
        print(f"   📊 Bucket distribution: {np.bincount(annotator.bucket_assignments)}")
        print(f"   🎯 CLIP-GT correlation: {np.corrcoef(annotator.clip_scores, dataset.labels)[0,1]:.3f}")
        
    except Exception as e:
        print(f"   ❌ CLIP initialization failed: {e}")
        print("   💡 Make sure you have CLIP installed: pip install git+https://github.com/openai/CLIP.git")
        return None
    
    # Run annotation session
    print(f"\n🚀 Running annotation session (max {max_comparisons} comparisons)...")
    results = annotator.run_annotation_session(max_comparisons)
    
    # Calculate final metrics
    if 'final_ranking' in results:
        final_ranking = results['final_ranking']
        metrics = calculate_ranking_metrics(final_ranking, dataset.labels)
        
        print(f"\n📊 Final Results:")
        print(f"   Total comparisons: {results['total_comparisons']}")
        print(f"   Human queries: {results['human_queries']}")
        print(f"   Auto decisions: {results['auto_decisions']}")
        print(f"   Automation rate: {results['automation_rate']:.1%}")
        print(f"   Spearman correlation: {metrics['spearman_correlation']:.3f}")
        print(f"   Kendall correlation: {metrics['kendall_correlation']:.3f}")
        print(f"   NDCG@10: {metrics['ndcg_at_10']:.3f}")
        
        # Show top 10 ranked faces
        print(f"\n🏆 Top 10 Oldest Faces (by EZ-Sort):")
        for i, idx in enumerate(final_ranking[:10]):
            age = dataset.labels[idx]
            img_name = dataset.image_paths[idx]
            print(f"   {i+1:2d}. {img_name} - Age: {age}")
    
    return results, annotator, dataset


def create_visualization(results, annotator, dataset, output_dir="results"):
    """Create visualization of results"""
    
    print(f"\n📊 Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Results dashboard
    visualize_results(results, dataset, os.path.join(output_dir, "dashboard.html"))
    
    # 2. Age distribution vs ranking
    if 'final_ranking' in results:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Original age distribution
        plt.subplot(2, 2, 1)
        plt.hist(dataset.labels, bins=20, alpha=0.7, color='blue')
        plt.title('Original Age Distribution')
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        
        # Plot 2: Ranked ages
        plt.subplot(2, 2, 2)
        ranked_ages = [dataset.labels[idx] for idx in results['final_ranking']]
        plt.plot(range(len(ranked_ages)), ranked_ages, 'ro-', alpha=0.7)
        plt.title('EZ-Sort Ranking vs True Age')
        plt.xlabel('Rank Position')
        plt.ylabel('True Age')
        
        # Plot 3: CLIP vs Ground Truth
        plt.subplot(2, 2, 3)
        plt.scatter(annotator.clip_scores, dataset.labels, alpha=0.6)
        plt.xlabel('CLIP Score')
        plt.ylabel('True Age')
        plt.title('CLIP Pre-ordering vs Ground Truth')
        
        # Add correlation line
        z = np.polyfit(annotator.clip_scores, dataset.labels, 1)
        p = np.poly1d(z)
        plt.plot(annotator.clip_scores, p(annotator.clip_scores), "r--", alpha=0.8)
        
        # Plot 4: Uncertainty distribution
        plt.subplot(2, 2, 4)
        if results['comparisons']:
            uncertainties = [c['uncertainty'] for c in results['comparisons']]
            human_uncert = [c['uncertainty'] for c in results['comparisons'] if c['type'] == 'human']
            auto_uncert = [c['uncertainty'] for c in results['comparisons'] if c['type'] == 'auto']
            
            plt.hist(human_uncert, bins=15, alpha=0.7, label='Human Queries', color='red')
            plt.hist(auto_uncert, bins=15, alpha=0.7, label='Auto Decisions', color='blue')
            plt.xlabel('Uncertainty Score')
            plt.ylabel('Frequency')
            plt.title('Uncertainty Distribution by Query Type')
            plt.legend()
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, "analysis.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Analysis plots saved to {viz_path}")
    
    # Export results
    export_results_to_csv(results, dataset, output_dir)
    
    print(f"   💾 All results exported to {output_dir}/")


def main():
    """Main example function"""
    
    print("🎯 EZ-Sort Face Age Estimation Example")
    print("=" * 40)
    
    # Parameters
    n_faces = 50
    max_comparisons = 80
    
    # Step 1: Create sample dataset
    csv_path, image_dir = create_sample_face_dataset(n_faces)
    
    # Step 2: Run annotation
    result = run_face_age_annotation(csv_path, image_dir, max_comparisons)
    
    if result is not None:
        results, annotator, dataset = result
        
        # Step 3: Create visualizations
        create_visualization(results, annotator, dataset)
        
        print(f"\n✅ Face age estimation example completed!")
        print(f"\n💡 Next steps:")
        print(f"   • Check the results/ directory for outputs")
        print(f"   • Try the web interface: streamlit run ez_sort_web.py")
        print(f"   • Modify the configuration in EZSortConfig for your needs")
        print(f"   • Use your own face dataset following the same CSV format")
    
    else:
        print(f"\n❌ Example failed - please check CLIP installation")


if __name__ == "__main__":
    main()


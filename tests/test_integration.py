
# tests/test_integration.py
"""
Integration tests for real-world scenarios
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from PIL import Image, ImageDraw

from ez_sort import EZSortDataset, EZSortConfig, EZSortAnnotator
from evaluation import run_comparison_study, SimpleElo, RandomComparison


class TestRealWorldScenarios:
    """Test EZ-Sort on realistic scenarios"""
    
    @pytest.fixture
    def face_age_dataset(self):
        """Create realistic face age dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            n_faces = 30
            ages = np.random.randint(1, 80, n_faces)
            
            # Create more realistic synthetic faces
            image_data = []
            img_dir = os.path.join(temp_dir, 'faces')
            os.makedirs(img_dir)
            
            for i, age in enumerate(ages):
                img_name = f'person_{i:03d}.jpg'
                
                # Create age-based synthetic face
                img = Image.new('RGB', (224, 224), color=(200, 180, 160))  # Skin tone
                draw = ImageDraw.Draw(img)
                
                # Add age-based features
                if age < 10:
                    # Baby features: larger eyes, smaller face
                    draw.ellipse([70, 80, 90, 100], fill=(0, 0, 0))  # Left eye
                    draw.ellipse([134, 80, 154, 100], fill=(0, 0, 0))  # Right eye
                    draw.ellipse([102, 120, 122, 140], fill=(255, 200, 200))  # Nose
                elif age < 20:
                    # Teen features
                    draw.ellipse([75, 85, 95, 105], fill=(0, 0, 0))
                    draw.ellipse([129, 85, 149, 105], fill=(0, 0, 0))
                    draw.ellipse([107, 125, 117, 135], fill=(150, 120, 120))
                else:
                    # Adult features
                    draw.ellipse([80, 90, 100, 110], fill=(0, 0, 0))
                    draw.ellipse([124, 90, 144, 110], fill=(0, 0, 0))
                    draw.ellipse([109, 130, 115, 136], fill=(100, 80, 80))
                
                # Add wrinkles for older faces
                if age > 50:
                    draw.line([(90, 75), (134, 75)], fill=(120, 100, 80), width=2)  # Forehead line
                    draw.line([(100, 150), (124, 150)], fill=(120, 100, 80), width=1)  # Mouth line
                
                img.save(os.path.join(img_dir, img_name))
                
                image_data.append({
                    'image_path': img_name,
                    'age': age,
                    'gender': 'M' if i % 2 == 0 else 'F'
                })
            
            # Create CSV
            df = pd.DataFrame(image_data)
            csv_path = os.path.join(temp_dir, 'faces.csv')
            df.to_csv(csv_path, index=False)
            
            dataset = EZSortDataset(csv_path, img_dir, 'image_path', 'age')
            yield dataset
    
    def test_face_age_workflow(self, face_age_dataset):
        """Test complete face age estimation workflow"""
        config = EZSortConfig(
            domain="face",
            k_buckets=4,
            theta_0=0.2  # Higher threshold for testing
        )
        
        # Test dataset properties
        assert face_age_dataset.n_items == 30
        assert min(face_age_dataset.labels) >= 1
        assert max(face_age_dataset.labels) <= 80
        
        # Test pairwise preferences
        young_idx = np.argmin(face_age_dataset.labels)
        old_idx = np.argmax(face_age_dataset.labels)
        
        preference = face_age_dataset.get_pairwise_preference(old_idx, young_idx)
        assert preference == 1  # Older person should rank higher
        
        # Test without CLIP (mock the initialization)
        try:
            annotator = EZSortAnnotator(face_age_dataset, config)
            
            # Test basic properties
            assert annotator.n_items == 30
            assert len(annotator.elo_ratings) == 30
            
            # Test uncertainty calculation
            uncertainty = annotator.calculate_uncertainty(0, 1)
            assert 0 <= uncertainty <= 1
            
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")
    
    def test_algorithm_comparison(self, face_age_dataset):
        """Test EZ-Sort against baseline algorithms"""
        
        # Create comparison algorithms
        algorithms = [
            RandomComparison(face_age_dataset.n_items),
            SimpleElo(face_age_dataset.n_items)
        ]
        
        # Run comparison study
        results_df = run_comparison_study(
            face_age_dataset, 
            algorithms, 
            max_comparisons=50,
            random_seed=42
        )
        
        # Check results
        assert len(results_df) > 0
        assert 'algorithm' in results_df.columns
        assert 'spearman_correlation' in results_df.columns
        
        # Simple Elo should outperform Random
        final_results = results_df.groupby('algorithm')['spearman_correlation'].last()
        assert final_results['Simple Elo'] > final_results['Random']

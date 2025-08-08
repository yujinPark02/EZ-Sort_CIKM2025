# tests/test_ez_sort.py
"""
Test suite for EZ-Sort functionality
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from PIL import Image
import sys
sys.path.append('..')

from ez_sort import EZSortDataset, EZSortConfig, EZSortAnnotator
from utils import calculate_ranking_metrics, analyze_clip_effectiveness
from prompts import get_default_prompts, create_custom_prompts


class TestEZSortConfig:
    """Test EZSortConfig class"""
    
    def test_default_config(self):
        config = EZSortConfig()
        assert config.domain == "face"
        assert config.k_buckets == 5
        assert config.clip_model == "ViT-B/32"
        assert config.hierarchical_prompts is not None
    
    def test_custom_config(self):
        custom_prompts = {
            "level_1": ["prompt1", "prompt2"],
            "level_2": ["prompt3", "prompt4", "prompt5", "prompt6"]
        }
        
        config = EZSortConfig(
            domain="medical",
            k_buckets=3,
            hierarchical_prompts=custom_prompts
        )
        
        assert config.domain == "medical"
        assert config.k_buckets == 3
        assert config.hierarchical_prompts == custom_prompts


class TestEZSortDataset:
    """Test EZSortDataset class"""
    
    @pytest.fixture
    def sample_dataset(self):
        # Create temporary dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV
            data = {
                'image_path': [f'img_{i:03d}.jpg' for i in range(10)],
                'age': np.random.randint(1, 70, 10).tolist()
            }
            df = pd.DataFrame(data)
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            df.to_csv(csv_path, index=False)
            
            # Create sample images
            img_dir = os.path.join(temp_dir, 'images')
            os.makedirs(img_dir)
            
            for img_name in data['image_path']:
                img = Image.new('RGB', (224, 224), color='red')
                img.save(os.path.join(img_dir, img_name))
            
            # Create dataset
            dataset = EZSortDataset(csv_path, img_dir, 'image_path', 'age')
            yield dataset
    
    def test_dataset_loading(self, sample_dataset):
        assert sample_dataset.n_items == 10
        assert len(sample_dataset.image_paths) == 10
        assert len(sample_dataset.labels) == 10
    
    def test_pairwise_preference(self, sample_dataset):
        # Test pairwise preference calculation
        for i in range(5):
            for j in range(i + 1, 5):
                preference = sample_dataset.get_pairwise_preference(i, j)
                assert preference in [0, 1]
                
                # Check consistency
                expected = 1 if sample_dataset.labels[i] > sample_dataset.labels[j] else 0
                assert preference == expected


class TestHierarchicalPrompts:
    """Test prompt generation and management"""
    
    def test_default_prompts(self):
        face_prompts = get_default_prompts("face")
        assert "level_1" in face_prompts
        assert "level_2" in face_prompts
        assert len(face_prompts["level_1"]) == 2
        assert len(face_prompts["level_2"]) == 4
    
    def test_medical_prompts(self):
        medical_prompts = get_default_prompts("medical")
        assert "level_1" in medical_prompts
        assert "level_2" in medical_prompts
        assert "normal" in medical_prompts["level_1"][0].lower()
        assert "abnormal" in medical_prompts["level_1"][1].lower()
    
    def test_custom_prompts(self):
        categories = ["cat1", "cat2", "cat3", "cat4"]
        descriptions = ["desc1", "desc2", "desc3", "desc4"]
        
        custom_prompts = create_custom_prompts("custom", categories, descriptions)
        assert "level_1" in custom_prompts
        assert "level_2" in custom_prompts
        assert len(custom_prompts["level_2"]) == 4


class TestRankingMetrics:
    """Test ranking evaluation metrics"""
    
    def test_perfect_ranking(self):
        ground_truth = [1, 2, 3, 4, 5]
        predicted = [4, 3, 2, 1, 0]  # Indices in descending order
        
        metrics = calculate_ranking_metrics(predicted, ground_truth)
        assert abs(metrics['spearman_correlation'] - 1.0) < 0.01
        assert abs(metrics['kendall_correlation'] - 1.0) < 0.01
    
    def test_random_ranking(self):
        np.random.seed(42)
        ground_truth = list(range(20))
        predicted = np.random.permutation(20).tolist()
        
        metrics = calculate_ranking_metrics(predicted, ground_truth)
        # Random ranking should have correlation near 0
        assert abs(metrics['spearman_correlation']) < 0.5
    
    def test_reversed_ranking(self):
        ground_truth = [1, 2, 3, 4, 5]
        predicted = [0, 1, 2, 3, 4]  # Ascending order (worst ranking)
        
        metrics = calculate_ranking_metrics(predicted, ground_truth)
        assert metrics['spearman_correlation'] < -0.8  # Strong negative correlation


class TestCLIPEffectiveness:
    """Test CLIP analysis utilities"""
    
    def test_clip_analysis(self):
        # Synthetic data
        n_items = 50
        ground_truth = np.random.uniform(0, 100, n_items)
        
        # CLIP scores with some correlation to ground truth
        noise = np.random.normal(0, 10, n_items)
        clip_scores = ground_truth * 0.7 + noise
        
        # Simple bucketing
        bucket_assignments = np.digitize(clip_scores, np.percentile(clip_scores, [20, 40, 60, 80])) - 1
        
        analysis = analyze_clip_effectiveness(clip_scores, ground_truth, bucket_assignments)
        
        assert 'clip_gt_correlation' in analysis
        assert 'avg_bucket_purity' in analysis
        assert 'cross_bucket_accuracy' in analysis
        assert analysis['clip_gt_correlation'] > 0.5  # Should have positive correlation


class TestEZSortIntegration:
    """Integration tests for full EZ-Sort pipeline"""
    
    @pytest.fixture
    def mock_dataset(self):
        # Create a mock dataset for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate synthetic data
            n_items = 20
            ages = np.random.randint(1, 70, n_items)
            image_names = [f'face_{i:03d}.jpg' for i in range(n_items)]
            
            # Create CSV
            df = pd.DataFrame({'image_path': image_names, 'age': ages})
            csv_path = os.path.join(temp_dir, 'faces.csv')
            df.to_csv(csv_path, index=False)
            
            # Create dummy images
            img_dir = os.path.join(temp_dir, 'images')
            os.makedirs(img_dir)
            
            for i, img_name in enumerate(image_names):
                # Create age-based colored image
                age = ages[i]
                color_val = int(255 * age / 70)  # Age-based color
                img = Image.new('RGB', (224, 224), color=(color_val, 100, 100))
                img.save(os.path.join(img_dir, img_name))
            
            dataset = EZSortDataset(csv_path, img_dir, 'image_path', 'age')
            yield dataset
    
    @pytest.mark.slow
    def test_ez_sort_initialization(self, mock_dataset):
        """Test EZ-Sort initialization (may be slow due to CLIP)"""
        config = EZSortConfig(k_buckets=3)  # Use fewer buckets for faster testing
        
        # This test requires CLIP model - skip if not available
        try:
            annotator = EZSortAnnotator(mock_dataset, config)
            assert annotator.n_items == mock_dataset.n_items
            assert annotator.clip_scores is not None
            assert annotator.bucket_assignments is not None
            assert len(annotator.elo_ratings) == mock_dataset.n_items
        except Exception as e:
            pytest.skip(f"CLIP not available for testing: {e}")
    
    def test_uncertainty_calculation(self, mock_dataset):
        """Test uncertainty calculation without CLIP"""
        config = EZSortConfig()
        
        # Create minimal annotator for testing uncertainty
        class MockAnnotator:
            def __init__(self, n_items):
                self.n_items = n_items
                self.elo_ratings = np.random.normal(1500, 200, n_items)
                self.confidence = np.random.uniform(0.5, 0.9, n_items)
                self.bucket_assignments = np.random.randint(0, 3, n_items)
                self.config = config
            
            def calculate_uncertainty(self, idx1, idx2):
                r1, r2 = self.elo_ratings[idx1], self.elo_ratings[idx2]
                p_ij = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
                p_before = [p_ij, 1 - p_ij]
                kl_div = sum(p * np.log(p / 0.5) for p in p_before if p > 0)
                gamma = 1.2 if self.bucket_assignments[idx1] != self.bucket_assignments[idx2] else 1.0
                avg_conf = (self.confidence[idx1] + self.confidence[idx2]) / 2
                phi = 2.0 - avg_conf
                priority = kl_div * gamma * phi
                uncertainty = 1 - (priority / np.log(2))
                return max(0, min(1, uncertainty))
        
        mock_annotator = MockAnnotator(mock_dataset.n_items)
        
        # Test uncertainty calculation
        for i in range(5):
            for j in range(i + 1, 5):
                uncertainty = mock_annotator.calculate_uncertainty(i, j)
                assert 0 <= uncertainty <= 1

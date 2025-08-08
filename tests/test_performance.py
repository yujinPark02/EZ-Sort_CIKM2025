
# tests/test_performance.py
"""
Performance and scalability tests
"""

import pytest
import time
import numpy as np
from ez_sort import EZSortConfig
from utils import calculate_ranking_metrics


class TestPerformance:
    """Test performance characteristics"""
    
    def test_ranking_metrics_performance(self):
        """Test performance of ranking metrics calculation"""
        
        # Large ranking test
        n_items = 1000
        ground_truth = np.random.uniform(0, 100, n_items)
        predicted_ranking = np.random.permutation(n_items).tolist()
        
        start_time = time.time()
        metrics = calculate_ranking_metrics(predicted_ranking, ground_truth)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max
        assert 'spearman_correlation' in metrics
    
    def test_config_creation_performance(self):
        """Test config creation performance"""
        
        start_time = time.time()
        for _ in range(100):
            config = EZSortConfig()
        elapsed = time.time() - start_time
        
        # Should be fast
        assert elapsed < 1.0  # 1 second for 100 configs
    
    def test_memory_usage(self):
        """Test memory usage with large arrays"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large arrays (simulating large dataset)
        n_items = 10000
        elo_ratings = np.random.normal(1500, 200, n_items)
        clip_scores = np.random.uniform(0, 1, n_items)
        confidences = np.random.uniform(0.5, 0.9, n_items)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory (less than 100MB for 10k items)
        assert memory_increase < 100


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# evaluation.py
"""
Evaluation utilities for comparing EZ-Sort with other methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import time
from abc import ABC, abstractmethod


class ComparisonAlgorithm(ABC):
    """Base class for comparison algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.comparison_count = 0
        self.start_time = None
    
    @abstractmethod
    def compare_and_update(self, idx1: int, idx2: int, preference: int) -> None:
        """Update algorithm state with comparison result"""
        pass
    
    @abstractmethod
    def get_ranking(self) -> List[int]:
        """Get current ranking"""
        pass
    
    def start_timing(self):
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


class RandomComparison(ComparisonAlgorithm):
    """Random baseline for comparison"""
    
    def __init__(self, n_items: int):
        super().__init__("Random")
        self.n_items = n_items
        self.scores = np.random.randn(n_items)
    
    def compare_and_update(self, idx1: int, idx2: int, preference: int) -> None:
        self.comparison_count += 1
        # Random doesn't learn from comparisons
        pass
    
    def get_ranking(self) -> List[int]:
        return np.argsort(self.scores)[::-1].tolist()


class SimpleElo(ComparisonAlgorithm):
    """Simple Elo rating system"""
    
    def __init__(self, n_items: int, k: int = 32):
        super().__init__("Simple Elo")
        self.n_items = n_items
        self.k = k
        self.ratings = np.full(n_items, 1500.0)
    
    def compare_and_update(self, idx1: int, idx2: int, preference: int) -> None:
        self.comparison_count += 1
        
        r1, r2 = self.ratings[idx1], self.ratings[idx2]
        e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        e2 = 1.0 / (1.0 + 10 ** ((r1 - r2) / 400))
        
        s1, s2 = float(preference), 1.0 - float(preference)
        
        self.ratings[idx1] += self.k * (s1 - e1)
        self.ratings[idx2] += self.k * (s2 - e2)
    
    def get_ranking(self) -> List[int]:
        return np.argsort(self.ratings)[::-1].tolist()


def run_comparison_study(dataset: Any,
                        algorithms: List[ComparisonAlgorithm],
                        max_comparisons: int = 100,
                        random_seed: int = 42) -> pd.DataFrame:
    """
    Run comparative study between different algorithms
    
    Args:
        dataset: Dataset with get_pairwise_preference method
        algorithms: List of algorithms to compare
        max_comparisons: Maximum comparisons per algorithm
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with results for each algorithm
    """
    
    np.random.seed(random_seed)
    
    # Generate comparison pairs
    n_items = min(dataset.n_items, 50)  # Limit for computational efficiency
    indices = list(range(n_items))
    
    pairs = []
    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            pairs.append((i, j))
    
    # Randomly sample pairs
    if len(pairs) > max_comparisons:
        selected_pairs = np.random.choice(len(pairs), max_comparisons, replace=False)
        pairs = [pairs[i] for i in selected_pairs]
    
    results = []
    
    for algorithm in algorithms:
        print(f"Running {algorithm.name}...")
        algorithm.start_timing()
        
        # Run comparisons
        for step, (idx1, idx2) in enumerate(pairs):
            preference = dataset.get_pairwise_preference(idx1, idx2)
            algorithm.compare_and_update(idx1, idx2, preference)
            
            # Record intermediate results
            if (step + 1) % 10 == 0 or step == len(pairs) - 1:
                current_ranking = algorithm.get_ranking()
                
                # Calculate accuracy (simplified)
                if hasattr(dataset, 'labels'):
                    ground_truth = [dataset.labels[i] for i in range(n_items)]
                    metrics = calculate_ranking_metrics(current_ranking, ground_truth)
                    spearman_corr = metrics['spearman_correlation']
                else:
                    spearman_corr = 0.0
                
                results.append({
                    'algorithm': algorithm.name,
                    'step': step + 1,
                    'comparisons': algorithm.comparison_count,
                    'spearman_correlation': spearman_corr,
                    'elapsed_time': algorithm.get_elapsed_time()
                })
    
    return pd.DataFrame(results)


def create_comparison_plot(results_df: pd.DataFrame, 
                          metric: str = 'spearman_correlation',
                          save_path: str = None) -> None:
    """Create comparison plot between algorithms"""
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    for algorithm in results_df['algorithm'].unique():
        alg_data = results_df[results_df['algorithm'] == algorithm]
        plt.plot(alg_data['step'], alg_data[metric], 
                marker='o', label=algorithm, linewidth=2)
    
    plt.xlabel('Number of Comparisons')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to {save_path}")
    else:
        plt.show()
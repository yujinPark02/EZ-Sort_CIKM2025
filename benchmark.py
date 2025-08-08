# benchmark.py
"""
Benchmarking script for EZ-Sort performance evaluation
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import psutil
import os
from dataclasses import dataclass

from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig
from evaluation import run_comparison_study, SimpleElo, RandomComparison


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    algorithm: str
    dataset_size: int
    comparisons: int
    accuracy: float
    time_taken: float
    memory_used: float
    automation_rate: float


class EZSortBenchmark:
    """Comprehensive benchmarking suite for EZ-Sort"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def create_synthetic_dataset(self, n_items: int, domain: str = "face") -> EZSortDataset:
        """Create synthetic dataset for benchmarking"""
        
        # Create temporary files
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic data
        if domain == "face":
            labels = np.random.randint(1, 80, n_items)  # Ages
            label_col = "age"
        elif domain == "quality":
            labels = np.random.uniform(0, 10, n_items)  # Quality scores
            label_col = "quality"
        else:
            labels = np.random.uniform(0, 100, n_items)  # Generic scores
            label_col = "score"
        
        # Create CSV
        df = pd.DataFrame({
            'image_path': [f'img_{i:04d}.jpg' for i in range(n_items)],
            label_col: labels
        })
        
        csv_path = os.path.join(temp_dir, 'benchmark_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Create dummy images directory (we won't actually load images for benchmarking)
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(img_dir)
        
        dataset = EZSortDataset(csv_path, img_dir, 'image_path', label_col)
        return dataset
    
    def benchmark_scaling(self, sizes: List[int] = None) -> None:
        """Benchmark EZ-Sort scaling with dataset size"""
        
        if sizes is None:
            sizes = [50, 100, 200, 500, 1000]
        
        print("📊 Running scaling benchmark...")
        
        for size in sizes:
            print(f"\n📈 Testing with {size} items...")
            
            # Create dataset
            dataset = self.create_synthetic_dataset(size)
            
            # Benchmark different algorithms
            max_comparisons = min(size * 2, 500)  # Reasonable limit
            
            algorithms = [
                ("EZ-Sort", lambda: self._benchmark_ez_sort(dataset, max_comparisons)),
                ("Simple Elo", lambda: self._benchmark_simple_elo(dataset, max_comparisons)),
                ("Random", lambda: self._benchmark_random(dataset, max_comparisons))
            ]
            
            for alg_name, benchmark_func in algorithms:
                try:
                    result = benchmark_func()
                    result.algorithm = alg_name
                    result.dataset_size = size
                    self.results.append(result)
                    
                    print(f"  ✅ {alg_name}: {result.accuracy:.3f} accuracy, {result.time_taken:.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ {alg_name} failed: {e}")
    
    def _benchmark_ez_sort(self, dataset: EZSortDataset, max_comparisons: int) -> BenchmarkResult:
        """Benchmark EZ-Sort specifically"""
        
        config = EZSortConfig(k_buckets=min(5, dataset.n_items // 10))
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        
        # Mock CLIP results to avoid actual model loading
        clip_scores = np.random.uniform(0, 1, dataset.n_items)
        confidence = np.random.uniform(0.5, 0.9, dataset.n_items)
        bucket_assignments = np.digitize(clip_scores, np.percentile(clip_scores, [20, 40, 60, 80])) - 1
        
        # Create annotator with mocked CLIP
        annotator = EZSortAnnotator(dataset, config)
        annotator.clip_scores = clip_scores
        annotator.confidence = confidence
        annotator.bucket_assignments = bucket_assignments
        annotator.elo_ratings = annotator._initialize_elo_ratings()
        
        # Run annotation
        results = annotator.run_annotation_session(max_comparisons)
        
        end_time = time.time()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate accuracy
        if 'final_ranking' in results:
            from utils import calculate_ranking_metrics
            metrics = calculate_ranking_metrics(results['final_ranking'], dataset.labels)
            accuracy = metrics['spearman_correlation']
        else:
            accuracy = 0.0
        
        return BenchmarkResult(
            algorithm="EZ-Sort",
            dataset_size=dataset.n_items,
            comparisons=results.get('total_comparisons', 0),
            accuracy=accuracy,
            time_taken=end_time - start_time,
            memory_used=memory_after - memory_before,
            automation_rate=results.get('automation_rate', 0)
        )
    
    def _benchmark_simple_elo(self, dataset: EZSortDataset, max_comparisons: int) -> BenchmarkResult:
        """Benchmark Simple Elo"""
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        elo = SimpleElo(dataset.n_items)
        
        # Run comparisons
        comparison_count = 0
        for i in range(dataset.n_items - 1):
            for j in range(i + 1, dataset.n_items):
                if comparison_count >= max_comparisons:
                    break
                
                preference = dataset.get_pairwise_preference(i, j)
                elo.compare_and_update(i, j, preference)
                comparison_count += 1
            
            if comparison_count >= max_comparisons:
                break
        
        ranking = elo.get_ranking()
        end_time = time.time()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Calculate accuracy
        from utils import calculate_ranking_metrics
        metrics = calculate_ranking_metrics(ranking, dataset.labels)
        accuracy = metrics['spearman_correlation']
        
        return BenchmarkResult(
            algorithm="Simple Elo",
            dataset_size=dataset.n_items,
            comparisons=comparison_count,
            accuracy=accuracy,
            time_taken=end_time - start_time,
            memory_used=memory_after - memory_before,
            automation_rate=0.0  # No automation
        )
    
    def _benchmark_random(self, dataset: EZSortDataset, max_comparisons: int) -> BenchmarkResult:
        """Benchmark Random baseline"""
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        random_alg = RandomComparison(dataset.n_items)
        ranking = random_alg.get_ranking()
        
        end_time = time.time()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Calculate accuracy
        from utils import calculate_ranking_metrics
        metrics = calculate_ranking_metrics(ranking, dataset.labels)
        accuracy = metrics['spearman_correlation']
        
        return BenchmarkResult(
            algorithm="Random",
            dataset_size=dataset.n_items,
            comparisons=0,  # No actual comparisons
            accuracy=accuracy,
            time_taken=end_time - start_time,
            memory_used=memory_after - memory_before,
            automation_rate=1.0  # Fully automated (but random)
        )
    
    def create_visualizations(self) -> None:
        """Create benchmark visualization plots"""
        
        if not self.results:
            print("❌ No benchmark results to visualize")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm,
                'Dataset Size': r.dataset_size,
                'Accuracy': r.accuracy,
                'Time (s)': r.time_taken,
                'Memory (MB)': r.memory_used,
                'Comparisons': r.comparisons,
                'Automation Rate': r.automation_rate
            }
            for r in self.results
        ])
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EZ-Sort Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy vs Dataset Size
        for algorithm in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == algorithm]
            axes[0, 0].plot(alg_data['Dataset Size'], alg_data['Accuracy'], 
                          marker='o', label=algorithm, linewidth=2)
        
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Spearman Correlation')
        axes[0, 0].set_title('Accuracy vs Dataset Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time vs Dataset Size
        for algorithm in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == algorithm]
            axes[0, 1].plot(alg_data['Dataset Size'], alg_data['Time (s)'], 
                          marker='s', label=algorithm, linewidth=2)
        
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Execution Time vs Dataset Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Memory vs Dataset Size
        for algorithm in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == algorithm]
            axes[0, 2].plot(alg_data['Dataset Size'], alg_data['Memory (MB)'], 
                          marker='^', label=algorithm, linewidth=2)
        
        axes[0, 2].set_xlabel('Dataset Size')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].set_title('Memory Usage vs Dataset Size')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Comparisons vs Dataset Size
        for algorithm in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == algorithm]
            if alg_data['Comparisons'].sum() > 0:  # Skip if no comparisons
                axes[1, 0].plot(alg_data['Dataset Size'], alg_data['Comparisons'], 
                              marker='d', label=algorithm, linewidth=2)
        
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Number of Comparisons')
        axes[1, 0].set_title('Comparisons vs Dataset Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Efficiency (Accuracy/Time)
        df['Efficiency'] = df['Accuracy'] / (df['Time (s)'] + 0.001)  # Avoid division by zero
        
        for algorithm in df['Algorithm'].unique():
            alg_data = df[df['Algorithm'] == algorithm]
            axes[1, 1].plot(alg_data['Dataset Size'], alg_data['Efficiency'], 
                          marker='*', label=algorithm, linewidth=2)
        
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Efficiency (Accuracy/Time)')
        axes[1, 1].set_title('Efficiency vs Dataset Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Automation Rate (EZ-Sort only)
        ez_sort_data = df[df['Algorithm'] == 'EZ-Sort']
        if not ez_sort_data.empty:
            axes[1, 2].plot(ez_sort_data['Dataset Size'], ez_sort_data['Automation Rate'], 
                          marker='o', color='green', linewidth=2)
            axes[1, 2].set_xlabel('Dataset Size')
            axes[1, 2].set_ylabel('Automation Rate')
            axes[1, 2].set_title('EZ-Sort Automation Rate')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No EZ-Sort data', ha='center', va='center', 
                          transform=axes[1, 2].transAxes, fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'benchmark_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Benchmark plot saved to: {plot_path}")
    
    def save_results(self) -> None:
        """Save benchmark results to CSV"""
        
        if not self.results:
            print("❌ No results to save")
            return
        
        # Convert to DataFrame and save
        df = pd.DataFrame([
            {
                'algorithm': r.algorithm,
                'dataset_size': r.dataset_size,
                'comparisons': r.comparisons,
                'accuracy': r.accuracy,
                'time_taken': r.time_taken,
                'memory_used': r.memory_used,
                'automation_rate': r.automation_rate
            }
            for r in self.results
        ])
        
        csv_path = os.path.join(self.output_dir, 'benchmark_results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"💾 Benchmark results saved to: {csv_path}")
        
        # Print summary
        print(f"\n📊 Benchmark Summary:")
        print(f"   Total tests: {len(self.results)}")
        print(f"   Algorithms: {df['algorithm'].nunique()}")
        print(f"   Dataset sizes: {sorted(df['dataset_size'].unique())}")
        
        # Best performance summary
        for metric in ['accuracy', 'time_taken', 'automation_rate']:
            if metric == 'time_taken':
                best = df.loc[df[metric].idxmin()]
                print(f"   Fastest: {best['algorithm']} ({best[metric]:.2f}s)")
            else:
                best = df.loc[df[metric].idxmax()]
                print(f"   Best {metric}: {best['algorithm']} ({best[metric]:.3f})")


def main():
    """Main benchmark function"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║               🚀 EZ-Sort Benchmark Suite                   ║
    ║                                                           ║
    ║  This script benchmarks EZ-Sort performance against      ║
    ║  baseline algorithms across different dataset sizes.      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create benchmark suite
    benchmark = EZSortBenchmark()
    
    # Run scaling benchmark
    sizes = [20, 50, 100, 200]  # Reasonable sizes for benchmarking
    benchmark.benchmark_scaling(sizes)
    
    # Create visualizations
    benchmark.create_visualizations()
    
    # Save results
    benchmark.save_results()
    
    print(f"\n✅ Benchmarking completed!")
    print(f"📂 Results saved to: {benchmark.output_dir}/")


if __name__ == "__main__":
    main()
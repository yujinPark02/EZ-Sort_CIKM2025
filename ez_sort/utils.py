# utils.py
"""
Utility functions for EZ-Sort evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from typing import List, Dict, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def calculate_ranking_metrics(predicted_ranking: List[int], 
                            ground_truth_labels: List[float],
                            dataset_indices: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Calculate ranking accuracy metrics
    
    Args:
        predicted_ranking: List of indices in predicted order (best to worst)
        ground_truth_labels: Ground truth values for each item
        dataset_indices: Optional mapping if ranking uses subset of dataset
    
    Returns:
        Dictionary with Spearman, Kendall, and other metrics
    """
    
    if dataset_indices is None:
        dataset_indices = list(range(len(ground_truth_labels)))
    
    # Get ground truth ranking (descending order)
    true_ranking = np.argsort(ground_truth_labels)[::-1].tolist()
    
    # Calculate metrics
    spearman_corr, spearman_p = spearmanr(true_ranking, predicted_ranking)
    kendall_corr, kendall_p = kendalltau(true_ranking, predicted_ranking)
    
    # Mean Average Precision for top-k
    def calculate_map_at_k(k: int) -> float:
        if k > len(predicted_ranking):
            k = len(predicted_ranking)
        
        top_k_pred = set(predicted_ranking[:k])
        top_k_true = set(true_ranking[:k])
        
        precision_scores = []
        for i in range(1, k + 1):
            top_i_pred = set(predicted_ranking[:i])
            precision = len(top_i_pred.intersection(top_k_true)) / i
            precision_scores.append(precision)
        
        return np.mean(precision_scores)
    
    # NDCG calculation
    def calculate_ndcg(k: int) -> float:
        if k > len(predicted_ranking):
            k = len(predicted_ranking)
        
        # DCG
        dcg = 0
        for i in range(k):
            idx = predicted_ranking[i]
            relevance = ground_truth_labels[idx]
            dcg += relevance / np.log2(i + 2)
        
        # IDCG
        sorted_labels = sorted(ground_truth_labels, reverse=True)
        idcg = 0
        for i in range(k):
            relevance = sorted_labels[i]
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    metrics = {
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'kendall_correlation': kendall_corr,
        'kendall_p_value': kendall_p,
        'map_at_5': calculate_map_at_k(5),
        'map_at_10': calculate_map_at_k(10),
        'ndcg_at_5': calculate_ndcg(5),
        'ndcg_at_10': calculate_ndcg(10),
        'ranking_accuracy_top_10': len(set(predicted_ranking[:10]).intersection(set(true_ranking[:10]))) / 10
    }
    
    return metrics


def visualize_results(results: Dict[str, Any], 
                     dataset: Any,
                     save_path: Optional[str] = None) -> None:
    """
    Create comprehensive visualization of EZ-Sort results
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Comparison Types Over Time',
            'Uncertainty Distribution', 
            'Bucket Distribution',
            'Annotation Efficiency'
        ],
        specs=[[{"secondary_y": True}, {"type": "histogram"}],
               [{"type": "bar"}, {"secondary_y": True}]]
    )
    
    # 1. Comparison types over time
    if 'comparisons' in results and results['comparisons']:
        df_comp = pd.DataFrame(results['comparisons'])
        
        # Human vs Auto comparisons over time
        human_steps = df_comp[df_comp['type'] == 'human']['step'].tolist()
        auto_steps = df_comp[df_comp['type'] == 'auto']['step'].tolist()
        
        fig.add_trace(
            go.Scatter(x=human_steps, y=[1]*len(human_steps), 
                      mode='markers', name='Human Queries',
                      marker=dict(color='red', size=8)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=auto_steps, y=[0]*len(auto_steps),
                      mode='markers', name='Auto Decisions', 
                      marker=dict(color='blue', size=8)),
            row=1, col=1
        )
        
        # Uncertainty trend
        fig.add_trace(
            go.Scatter(x=df_comp['step'], y=df_comp['uncertainty'],
                      mode='lines', name='Uncertainty',
                      line=dict(color='orange')),
            row=1, col=1, secondary_y=True
        )
    
    # 2. Uncertainty distribution
    if 'comparisons' in results and results['comparisons']:
        uncertainties = [c['uncertainty'] for c in results['comparisons']]
        fig.add_trace(
            go.Histogram(x=uncertainties, nbinsx=20, name='Uncertainty Distribution'),
            row=1, col=2
        )
    
    # 3. Bucket distribution (if available)
    if hasattr(dataset, 'bucket_assignments'):
        bucket_counts = np.bincount(dataset.bucket_assignments)
        fig.add_trace(
            go.Bar(x=list(range(len(bucket_counts))), y=bucket_counts,
                  name='Bucket Distribution'),
            row=2, col=1
        )
    
    # 4. Efficiency metrics
    total_comps = results.get('human_queries', 0) + results.get('auto_decisions', 0)
    auto_rate = results.get('auto_decisions', 0) / total_comps * 100 if total_comps > 0 else 0
    
    fig.add_trace(
        go.Bar(x=['Human Queries', 'Auto Decisions'], 
               y=[results.get('human_queries', 0), results.get('auto_decisions', 0)],
               name='Query Types'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="EZ-Sort Annotation Results Dashboard",
        showlegend=True,
        height=800
    )
    
    fig.update_yaxes(title_text="Query Type", row=1, col=1)
    fig.update_yaxes(title_text="Uncertainty", secondary_y=True, row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=1)
    
    fig.update_xaxes(title_text="Uncertainty Score", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Bucket ID", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_xaxes(title_text="Query Type", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"📊 Visualization saved to {save_path}")
    else:
        fig.show()


def export_results_to_csv(results: Dict[str, Any], 
                         dataset: Any,
                         output_dir: str) -> None:
    """Export results to CSV files"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Comparisons
    if 'comparisons' in results:
        df_comp = pd.DataFrame(results['comparisons'])
        df_comp.to_csv(os.path.join(output_dir, 'comparisons.csv'), index=False)
    
    # 2. Final ranking
    if 'final_ranking' in results:
        ranking_data = []
        for rank, idx in enumerate(results['final_ranking']):
            ranking_data.append({
                'rank': rank + 1,
                'image_index': idx,
                'image_path': dataset.image_paths[idx] if hasattr(dataset, 'image_paths') else f'item_{idx}',
                'label': dataset.labels[idx] if hasattr(dataset, 'labels') else None
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking.to_csv(os.path.join(output_dir, 'final_ranking.csv'), index=False)
    
    # 3. Summary metrics
    metrics = {
        'total_comparisons': results.get('human_queries', 0) + results.get('auto_decisions', 0),
        'human_queries': results.get('human_queries', 0),
        'auto_decisions': results.get('auto_decisions', 0),
        'automation_rate': results.get('automation_rate', 0)
    }
    
    if 'comparisons' in results and results['comparisons']:
        uncertainties = [c['uncertainty'] for c in results['comparisons']]
        metrics.update({
            'avg_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'min_uncertainty': np.min(uncertainties),
            'max_uncertainty': np.max(uncertainties)
        })
    
    df_summary = pd.DataFrame([metrics])
    df_summary.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    print(f"📁 Results exported to {output_dir}/")


def analyze_clip_effectiveness(clip_scores: np.ndarray, 
                              ground_truth: np.ndarray,
                              bucket_assignments: np.ndarray) -> Dict[str, float]:
    """Analyze the effectiveness of CLIP pre-ordering"""
    
    # CLIP-GT correlation
    clip_gt_corr, _ = spearmanr(clip_scores, ground_truth)
    
    # Bucket purity (how well buckets separate ground truth)
    bucket_purities = []
    for bucket_id in np.unique(bucket_assignments):
        bucket_mask = bucket_assignments == bucket_id
        bucket_gt = ground_truth[bucket_mask]
        if len(bucket_gt) > 1:
            bucket_std = np.std(bucket_gt)
            bucket_purities.append(1.0 / (1.0 + bucket_std))  # Higher purity = lower std
        else:
            bucket_purities.append(1.0)
    
    avg_bucket_purity = np.mean(bucket_purities)
    
    # Cross-bucket ordering accuracy
    cross_bucket_accuracy = 0
    n_cross_bucket = 0
    
    for i in range(len(bucket_assignments)):
        for j in range(i + 1, len(bucket_assignments)):
            if bucket_assignments[i] != bucket_assignments[j]:
                n_cross_bucket += 1
                if (clip_scores[i] > clip_scores[j]) == (ground_truth[i] > ground_truth[j]):
                    cross_bucket_accuracy += 1
    
    cross_bucket_accuracy = cross_bucket_accuracy / n_cross_bucket if n_cross_bucket > 0 else 0
    
    return {
        'clip_gt_correlation': clip_gt_corr,
        'avg_bucket_purity': avg_bucket_purity,
        'cross_bucket_accuracy': cross_bucket_accuracy,
        'n_buckets': len(np.unique(bucket_assignments))
    }

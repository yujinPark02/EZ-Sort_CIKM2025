"""
EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting

Implementation of the EZ-Sort framework from the CIKM 2025 paper.
This tool enables efficient human annotation for pairwise ranking tasks.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import heapq
from scipy.stats import spearmanr, kendalltau


@dataclass
class EZSortConfig:
    """Configuration for EZ-Sort algorithm"""
    # CLIP parameters
    clip_model: str = "ViT-B/32"
    temperature: float = 0.1
    
    # Hierarchical prompts (default: face age estimation)
    domain: str = "face"
    range_description: str = "0-60+ years"
    hierarchical_prompts: Dict[str, List[str]] = None
    
    # Bucket parameters
    k_buckets: int = 5
    
    # Elo rating parameters
    elo_k: int = 32
    r_base_min: float = 1200.0
    r_base_max: float = 1800.0
    delta_b: float = 75.0
    
    # Uncertainty threshold parameters
    theta_0: float = 0.15
    alpha: float = 0.3
    beta: float = 0.9
    
    # Priority parameters
    gamma: float = 1.2  # Cross-bucket multiplier
    phi_base: float = 2.0  # Confidence penalty base
    
    def __post_init__(self):
        if self.hierarchical_prompts is None:
            # Default face age estimation prompts
            self.hierarchical_prompts = {
                "level_1": [
                    "a photograph of a baby or infant with rounded cheeks and large forehead",
                    "a photograph of a child or teenager with developing facial features"
                ],
                "level_2": [
                    "a photograph of a baby (0-2 years) with very soft facial features and chubby cheeks",
                    "a photograph of a young child (3-7 years) with childlike facial proportions",
                    "a photograph of a teenager (8-17 years) with adolescent facial features",
                    "a photograph of a young adult (18-35 years) with mature but youthful features"
                ],
                "level_3": [
                    "a photograph of a baby (0-1 years) with very soft and rounded facial features",
                    "a photograph of a toddler (2-4 years) with developing facial structure",
                    "a photograph of a child (5-9 years) with clear childlike features",
                    "a photograph of a pre-teen (10-13 years) with transitional facial features",
                    "a photograph of a teenager (14-18 years) with adolescent characteristics",
                    "a photograph of a young adult (19-30 years) with fully developed youthful features",
                    "a photograph of an adult (31-50 years) with mature facial characteristics",
                    "a photograph of an older adult (50+ years) with visible signs of aging"
                ]
            }


class EZSortDataset:
    """Dataset class for EZ-Sort"""
    
    def __init__(self, csv_path: str, image_dir: str, image_column: str = "image_path", 
                 label_column: str = "label", label_type: str = "continuous"):
        """
        Initialize dataset
        
        Args:
            csv_path: Path to CSV file with image paths and labels
            image_dir: Directory containing images
            image_column: Column name for image paths in CSV
            label_column: Column name for ground truth labels
            label_type: "continuous" or "categorical"
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.image_column = image_column
        self.label_column = label_column
        self.label_type = label_type
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df[image_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.n_items = len(self.df)
        
        # Precomputed CLIP results
        self.clip_scores = None
        self.clip_confidence = None
        self.bucket_assignments = None
        self.clip_metadata = {}
        
        print(f"Loaded dataset: {self.n_items} items from {csv_path}")
        
    def get_image_path(self, idx: int) -> str:
        """Get full path for image at index"""
        return os.path.join(self.image_dir, self.image_paths[idx])
        
    def get_pairwise_preference(self, idx1: int, idx2: int) -> int:
        """Get ground truth pairwise preference (for evaluation)"""
        if self.label_type == "continuous":
            return 1 if self.labels[idx1] > self.labels[idx2] else 0
        else:
            # For categorical, assume higher category = higher rank
            return 1 if self.labels[idx1] > self.labels[idx2] else 0


class HierarchicalCLIPClassifier:
    """Hierarchical CLIP-based classifier for zero-shot pre-ordering"""
    
    def __init__(self, config: EZSortConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(config.clip_model, device=self.device)
        
    def classify_hierarchical(self, dataset: EZSortDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform hierarchical CLIP classification
        
        Returns:
            clip_scores: Final CLIP-based scores for ranking
            confidence_scores: Confidence scores for each item
            bucket_assignments: Bucket assignments for each item
        """
        print("🧠 Running hierarchical CLIP classification...")
        
        # Load and preprocess images
        images = []
        valid_indices = []
        
        for i, img_path in enumerate([dataset.get_image_path(i) for i in range(dataset.n_items)]):
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(self.preprocess(image).unsqueeze(0).to(self.device))
                valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                
        if not images:
            raise RuntimeError("No valid images found")
            
        # Stack images
        image_batch = torch.cat(images, dim=0)
        
        # Hierarchical classification
        level_results = {}
        final_groups = np.zeros(len(valid_indices), dtype=int)
        overall_confidence = np.zeros(len(valid_indices))
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            current_groups = np.zeros(len(valid_indices), dtype=int)
            
            for level_name, prompts in self.config.hierarchical_prompts.items():
                level_num = int(level_name.split('_')[1])
                print(f"  Processing {level_name} with {len(prompts)} prompts...")
                
                # Encode text prompts
                text_tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T) / self.config.temperature
                probabilities = torch.softmax(similarities, dim=-1)
                
                # Get classifications and confidence
                classifications = torch.argmax(probabilities, dim=-1).cpu().numpy()
                confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
                
                # Update groups hierarchically
                group_offset = current_groups * len(prompts)
                current_groups = group_offset + classifications
                
                level_results[level_name] = {
                    'classifications': classifications,
                    'confidences': confidences,
                    'groups': current_groups.copy()
                }
                
                # Accumulate confidence (average across levels)
                overall_confidence += confidences / len(self.config.hierarchical_prompts)
        
        # Final group assignments and scoring
        final_groups = current_groups
        max_groups = np.max(final_groups) + 1
        
        # Convert groups to continuous scores (preserve order)
        clip_scores = final_groups.astype(float)
        # Add small random noise to break ties
        clip_scores += np.random.normal(0, 0.01, len(clip_scores))
        
        # Bucket assignments
        bucket_assignments = self._assign_buckets(final_groups, self.config.k_buckets)
        
        # Expand results to full dataset size (handle missing images)
        full_clip_scores = np.zeros(dataset.n_items)
        full_confidence = np.zeros(dataset.n_items)
        full_buckets = np.zeros(dataset.n_items, dtype=int)
        
        for i, valid_idx in enumerate(valid_indices):
            full_clip_scores[valid_idx] = clip_scores[i]
            full_confidence[valid_idx] = overall_confidence[i]
            full_buckets[valid_idx] = bucket_assignments[i]
        
        # Store metadata
        dataset.clip_metadata = {
            'level_results': level_results,
            'valid_indices': valid_indices,
            'n_levels': len(self.config.hierarchical_prompts)
        }
        
        print(f"  ✓ Classified {len(valid_indices)} images into {max_groups} groups")
        print(f"  ✓ Bucket distribution: {np.bincount(full_buckets)}")
        
        return full_clip_scores, full_confidence, full_buckets
    
    def _assign_buckets(self, groups: np.ndarray, k_buckets: int) -> np.ndarray:
        """Assign groups to k buckets uniformly"""
        max_group = np.max(groups)
        bucket_size = (max_group + 1) / k_buckets
        return np.floor(groups / bucket_size).astype(int)
        bucket_assignments = np.minimum(bucket_assignments, k_buckets - 1)
        return bucket_assignments


class EZSortAnnotator:
    """Main EZ-Sort algorithm implementation"""
    
    def __init__(self, dataset: EZSortDataset, config: EZSortConfig):
        self.dataset = dataset
        self.config = config
        self.n_items = dataset.n_items
        
        # Initialize CLIP classifier and perform pre-ordering
        self.clip_classifier = HierarchicalCLIPClassifier(config)
        clip_scores, confidence, buckets = self.clip_classifier.classify_hierarchical(dataset)
        
        # Store CLIP results
        self.clip_scores = clip_scores
        self.confidence = confidence
        self.bucket_assignments = buckets
        
        # Initialize Elo ratings
        self.elo_ratings = self._initialize_elo_ratings()
        
        # Tracking
        self.comparison_count = 0
        self.human_comparison_count = 0
        self.auto_comparison_count = 0
        self.comparison_history = []
        
        print(f"EZ-Sort initialized with {self.n_items} items, {config.k_buckets} buckets")
        
    def _initialize_elo_ratings(self) -> np.ndarray:
        """Initialize bucket-aware Elo ratings"""
        # Base ratings linearly distributed across buckets
        r_base = np.linspace(self.config.r_base_min, self.config.r_base_max, self.config.k_buckets)
        base_ratings = r_base[self.bucket_assignments]
        
        # CLIP score bonus
        clip_normalized = (self.clip_scores - np.min(self.clip_scores)) / (np.max(self.clip_scores) - np.min(self.clip_scores) + 1e-8)
        clip_bonus = clip_normalized * 200 - 100  # -100 to +100 bonus
        
        # Random noise with confidence weighting
        eta = np.random.uniform(-self.config.delta_b, self.config.delta_b, self.n_items)
        confidence_term = eta * (1.5 - self.confidence)
        
        ratings = base_ratings + clip_bonus + confidence_term
        return ratings
    
    def calculate_uncertainty(self, idx1: int, idx2: int) -> float:
        """Calculate uncertainty for a pair using KL divergence"""
        r1, r2 = self.elo_ratings[idx1], self.elo_ratings[idx2]
        
        # Elo prediction probability
        p_ij = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        
        # KL divergence from uniform
        p_before = [p_ij, 1 - p_ij]
        kl_div = sum(p * np.log(p / 0.5) for p in p_before if p > 0)
        
        # Priority factors
        gamma = self.config.gamma if self.bucket_assignments[idx1] != self.bucket_assignments[idx2] else 1.0
        avg_conf = (self.confidence[idx1] + self.confidence[idx2]) / 2
        phi = self.config.phi_base - avg_conf
        
        priority = kl_div * gamma * phi
        uncertainty = 1 - (priority / np.log(2))  # Normalize by max binary info gain
        
        return max(0, min(1, uncertainty))
    
    def should_query_human(self, idx1: int, idx2: int, current_step: int = 0, total_steps: int = 1000) -> bool:
        """Decide whether to query human for this pair"""
        uncertainty = self.calculate_uncertainty(idx1, idx2)
        
        # Adaptive threshold
        remaining_ratio = max(0, (total_steps - current_step) / total_steps)
        current_accuracy = 0.8  # Placeholder - would be computed from current ranking
        
        threshold = self.config.theta_0 * (1 + self.config.alpha * remaining_ratio) ** (self.config.beta / current_accuracy)
        
        return uncertainty >= threshold
    
    def update_elo(self, idx1: int, idx2: int, preference: int):
        """Update Elo ratings based on comparison result"""
        r1, r2 = self.elo_ratings[idx1], self.elo_ratings[idx2]
        
        # Expected scores
        e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        e2 = 1.0 / (1.0 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores (preference: 1 if idx1 > idx2, 0 otherwise)
        s1 = float(preference)
        s2 = 1.0 - s1
        
        # Update ratings
        self.elo_ratings[idx1] += self.config.elo_k * (s1 - e1)
        self.elo_ratings[idx2] += self.config.elo_k * (s2 - e2)
    
    def get_ranking(self) -> List[int]:
        """Get current ranking based on Elo scores"""
        return np.argsort(self.elo_ratings)[::-1].tolist()
    
    def run_annotation_session(self, max_comparisons: int, human_oracle_func=None) -> Dict[str, Any]:
        """
        Run annotation session with human-in-the-loop
        
        Args:
            max_comparisons: Maximum number of comparisons to perform
            human_oracle_func: Function to query human (idx1, idx2) -> preference
                             If None, uses ground truth for simulation
        """
        print(f"🚀 Starting EZ-Sort annotation session (max {max_comparisons} comparisons)")
        
        # Use ground truth oracle if no human function provided
        if human_oracle_func is None:
            human_oracle_func = self.dataset.get_pairwise_preference
        
        # Initialize merge sort queue with all possible pairs
        comparison_queue = []
        indices = list(range(self.n_items))
        
        # Add initial comparisons (simplified - would use proper MergeSort schedule)
        for i in range(len(indices) - 1):
            for j in range(i + 1, len(indices)):
                heapq.heappush(comparison_queue, (0, indices[i], indices[j]))  # Priority 0 for now
        
        results = {
            'comparisons': [],
            'human_queries': 0,
            'auto_decisions': 0,
            'ranking_history': []
        }
        
        for step in range(min(max_comparisons, len(comparison_queue))):
            if not comparison_queue:
                break
                
            # Get next pair (simplified - real implementation would follow MergeSort schedule)
            _, idx1, idx2 = heapq.heappop(comparison_queue)
            
            # Check if human query is needed
            if self.should_query_human(idx1, idx2, step, max_comparisons):
                # Human comparison
                preference = human_oracle_func(idx1, idx2)
                results['human_queries'] += 1
                self.human_comparison_count += 1
                query_type = "human"
            else:
                # Automatic comparison based on current Elo
                preference = 1 if self.elo_ratings[idx1] > self.elo_ratings[idx2] else 0
                results['auto_decisions'] += 1
                self.auto_comparison_count += 1
                query_type = "auto"
            
            # Update Elo ratings
            self.update_elo(idx1, idx2, preference)
            
            # Record comparison
            results['comparisons'].append({
                'step': step,
                'idx1': idx1,
                'idx2': idx2,
                'preference': preference,
                'type': query_type,
                'uncertainty': self.calculate_uncertainty(idx1, idx2)
            })
            
            # Record ranking every 10 steps
            if step % 10 == 0:
                current_ranking = self.get_ranking()
                results['ranking_history'].append({
                    'step': step,
                    'ranking': current_ranking.copy()
                })
        
        # Final ranking
        final_ranking = self.get_ranking()
        results['final_ranking'] = final_ranking
        
        # Calculate efficiency metrics
        total_comparisons = results['human_queries'] + results['auto_decisions']
        automation_rate = results['auto_decisions'] / total_comparisons if total_comparisons > 0 else 0
        
        print(f"✅ Annotation session completed:")
        print(f"  Total comparisons: {total_comparisons}")
        print(f"  Human queries: {results['human_queries']}")
        print(f"  Auto decisions: {results['auto_decisions']}")
        print(f"  Automation rate: {automation_rate:.1%}")
        
        results['total_comparisons'] = total_comparisons
        results['automation_rate'] = automation_rate
        
        return results


def main():
    """Example usage of EZ-Sort"""
    # Example configuration
    config = EZSortConfig(
        domain="face",
        range_description="0-60+ years",
        k_buckets=5
    )
    
    # Load dataset (example)
    # dataset = EZSortDataset(
    #     csv_path="data/face_dataset.csv",
    #     image_dir="data/images/",
    #     image_column="image_path",
    #     label_column="age"
    # )
    
    # Create EZ-Sort annotator
    # annotator = EZSortAnnotator(dataset, config)
    
    # Run annotation session
    # results = annotator.run_annotation_session(max_comparisons=100)
    
    print("EZ-Sort ready! Check the README for usage instructions.")


if __name__ == "__main__":
    main()
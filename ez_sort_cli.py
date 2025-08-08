#!/usr/bin/env python3
"""
EZ-Sort Command Line Interface
Provides command-line access to the EZ-Sort annotation framework.
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any
from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig


def load_config(config_path: str) -> EZSortConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert to EZSortConfig
        config = EZSortConfig(**config_dict)
        return config
    
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using default face age estimation configuration.")
        return EZSortConfig()


def simulate_annotation_session(annotator: EZSortAnnotator, max_comparisons: int, 
                              interactive: bool = False) -> Dict[str, Any]:
    """Run annotation session (simulation or interactive)"""
    
    print(f"🚀 Starting EZ-Sort annotation session...")
    print(f"   Dataset: {annotator.n_items} items")
    print(f"   Max comparisons: {max_comparisons}")
    print(f"   Mode: {'Interactive' if interactive else 'Simulation'}")
    
    if interactive:
        return run_interactive_session(annotator, max_comparisons)
    else:
        return run_simulation_session(annotator, max_comparisons)


def run_interactive_session(annotator: EZSortAnnotator, max_comparisons: int) -> Dict[str, Any]:
    """Run interactive annotation session with human input"""
    
    results = {
        'comparisons': [],
        'human_queries': 0,
        'auto_decisions': 0,
        'ranking_history': []
    }
    
    # Create simple comparison queue
    comparison_queue = []
    indices = list(range(min(annotator.n_items, 50)))  # Limit for CLI demo
    
    for i in range(len(indices) - 1):
        for j in range(i + 1, min(i + 5, len(indices))):  # Limit pairs per item
            comparison_queue.append((indices[i], indices[j]))
    
    print(f"\n📋 Total potential comparisons: {len(comparison_queue)}")
    print("=" * 60)
    
    for step in range(min(max_comparisons, len(comparison_queue))):
        idx1, idx2 = comparison_queue[step]
        
        # Check if human query is needed
        uncertainty = annotator.calculate_uncertainty(idx1, idx2)
        should_query = annotator.should_query_human(idx1, idx2, step, max_comparisons)
        
        if should_query:
            # Human comparison
            print(f"\n🤔 Comparison {step + 1}/{max_comparisons}")
            print(f"   Uncertainty: {uncertainty:.3f}")
            print(f"   Image A: {annotator.dataset.image_paths[idx1]}")
            print(f"   Image B: {annotator.dataset.image_paths[idx2]}")
            
            while True:
                try:
                    choice = input("\n   Which ranks higher? [A/B/S(kip)/Q(uit)]: ").strip().upper()
                    
                    if choice == 'A':
                        preference = 1
                        break
                    elif choice == 'B':
                        preference = 0
                        break
                    elif choice == 'S':
                        print("   ⏭️ Skipped")
                        continue
                    elif choice == 'Q':
                        print("   🛑 Annotation session terminated by user")
                        break
                    else:
                        print("   ❌ Invalid input. Please enter A, B, S, or Q.")
                        
                except KeyboardInterrupt:
                    print("\n   🛑 Annotation session interrupted")
                    choice = 'Q'
                    break
            
            if choice == 'Q':
                break
            elif choice == 'S':
                continue
            
            # Update Elo ratings
            annotator.update_elo(idx1, idx2, preference)
            results['human_queries'] += 1
            query_type = "human"
            
            print(f"   ✅ Recorded: {'A' if preference else 'B'} ranks higher")
        
        else:
            # Auto comparison
            preference = 1 if annotator.elo_ratings[idx1] > annotator.elo_ratings[idx2] else 0
            annotator.update_elo(idx1, idx2, preference)
            results['auto_decisions'] += 1
            query_type = "auto"
            
            print(f"\n🤖 Auto-comparison {step + 1}: Image {'A' if preference else 'B'} ranks higher (uncertainty: {uncertainty:.3f})")
        
        # Record comparison
        results['comparisons'].append({
            'step': step,
            'idx1': idx1,
            'idx2': idx2,
            'preference': preference,
            'type': query_type,
            'uncertainty': uncertainty
        })
        
        # Progress update every 10 steps
        if (step + 1) % 10 == 0:
            total = results['human_queries'] + results['auto_decisions']
            auto_rate = results['auto_decisions'] / total * 100
            print(f"\n📊 Progress: {step + 1}/{max_comparisons} | "
                  f"Human: {results['human_queries']} | "
                  f"Auto: {results['auto_decisions']} | "
                  f"Automation: {auto_rate:.1f}%")
    
    # Final ranking
    final_ranking = annotator.get_ranking()
    results['final_ranking'] = final_ranking
    
    return results


def run_simulation_session(annotator: EZSortAnnotator, max_comparisons: int) -> Dict[str, Any]:
    """Run simulation using ground truth for comparisons"""
    
    print("🔄 Running simulation with ground truth oracle...")
    
    # Use the built-in simulation method
    results = annotator.run_annotation_session(max_comparisons)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, dataset: EZSortDataset):
    """Save annotation results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results as JSON
    results_path = os.path.join(output_dir, "ez_sort_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparisons as CSV
    if results['comparisons']:
        comparisons_df = pd.DataFrame(results['comparisons'])
        comparisons_path = os.path.join(output_dir, "comparisons.csv")
        comparisons_df.to_csv(comparisons_path, index=False)
    
    # Save final ranking as CSV
    if 'final_ranking' in results:
        ranking_data = []
        for rank, idx in enumerate(results['final_ranking']):
            ranking_data.append({
                'rank': rank + 1,
                'image_index': idx,
                'image_path': dataset.image_paths[idx],
                'original_label': dataset.labels[idx],
                'final_elo_score': float(results.get('final_elo_scores', [0] * len(results['final_ranking']))[rank] if 'final_elo_scores' in results else 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_path = os.path.join(output_dir, "final_ranking.csv")
        ranking_df.to_csv(ranking_path, index=False)
    
    print(f"\n💾 Results saved to {output_dir}/")
    print(f"   - ez_sort_results.json (full results)")
    print(f"   - comparisons.csv (comparison history)")
    print(f"   - final_ranking.csv (final ranking)")


def print_summary(results: Dict[str, Any]):
    """Print session summary"""
    
    total_comparisons = results['human_queries'] + results['auto_decisions']
    automation_rate = results['auto_decisions'] / total_comparisons * 100 if total_comparisons > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 ANNOTATION SESSION SUMMARY")
    print("=" * 60)
    print(f"Total comparisons:     {total_comparisons}")
    print(f"Human queries:         {results['human_queries']}")
    print(f"Auto decisions:        {results['auto_decisions']}")
    print(f"Automation rate:       {automation_rate:.1f}%")
    
    if total_comparisons > 0:
        avg_uncertainty = np.mean([c['uncertainty'] for c in results['comparisons']])
        human_uncertainty = np.mean([c['uncertainty'] for c in results['comparisons'] if c['type'] == 'human'])
        auto_uncertainty = np.mean([c['uncertainty'] for c in results['comparisons'] if c['type'] == 'auto'])
        
        print(f"Average uncertainty:   {avg_uncertainty:.3f}")
        print(f"Human query uncert.:   {human_uncertainty:.3f}")
        print(f"Auto decision uncert.: {auto_uncertainty:.3f}")
    
    print("=" * 60)


def create_example_config():
    """Create example configuration file"""
    
    config_dict = {
        "clip_model": "ViT-B/32",
        "temperature": 0.1,
        "domain": "face",
        "range_description": "0-60+ years",
        "k_buckets": 5,
        "elo_k": 32,
        "r_base_min": 1200.0,
        "r_base_max": 1800.0,
        "delta_b": 75.0,
        "theta_0": 0.15,
        "alpha": 0.3,
        "beta": 0.9,
        "gamma": 1.2,
        "phi_base": 2.0,
        "hierarchical_prompts": {
            "level_1": [
                "a photograph of a baby or infant with rounded cheeks and large forehead",
                "a photograph of a child or teenager with developing facial features"
            ],
            "level_2": [
                "a photograph of a baby (0-2 years) with very soft facial features",
                "a photograph of a young child (3-7 years) with childlike proportions", 
                "a photograph of a teenager (8-17 years) with adolescent features",
                "a photograph of a young adult (18-35 years) with mature features"
            ],
            "level_3": [
                "a photograph of a baby (0-1 years) with very soft and rounded features",
                "a photograph of a toddler (2-4 years) with developing structure",
                "a photograph of a child (5-9 years) with clear childlike features",
                "a photograph of a pre-teen (10-13 years) with transitional features",
                "a photograph of a teenager (14-18 years) with adolescent characteristics",
                "a photograph of a young adult (19-30 years) with youthful features",
                "a photograph of an adult (31-50 years) with mature characteristics",
                "a photograph of an older adult (50+ years) with signs of aging"
            ]
        }
    }
    
    with open("config_face_age.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("📄 Created example configuration: config_face_age.json")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example configuration
  python ez_sort_cli.py --create-config
  
  # Run simulation with face age estimation
  python ez_sort_cli.py --csv data/faces.csv --images data/face_images/ --max-comparisons 100
  
  # Run interactive annotation
  python ez_sort_cli.py --csv data/faces.csv --images data/face_images/ --interactive --max-comparisons 50
  
  # Use custom configuration
  python ez_sort_cli.py --csv data/medical.csv --images data/medical_images/ --config config_medical.json
        """
    )
    
    # Main arguments
    parser.add_argument("--csv", type=str, help="Path to CSV file with image paths and labels")
    parser.add_argument("--images", type=str, help="Directory containing images")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    
    # Dataset columns
    parser.add_argument("--image-col", type=str, default="image_path", help="Column name for image paths")
    parser.add_argument("--label-col", type=str, default="label", help="Column name for labels")
    
    # Annotation parameters
    parser.add_argument("--max-comparisons", type=int, default=100, help="Maximum number of comparisons")
    parser.add_argument("--interactive", action="store_true", help="Run interactive annotation (vs simulation)")
    
    # Utility
    parser.add_argument("--create-config", action="store_true", help="Create example configuration file")
    
    args = parser.parse_args()
    
    # Create example config
    if args.create_config:
        create_example_config()
        return
    
    # Validate required arguments
    if not args.csv or not args.images:
        parser.error("Both --csv and --images are required (unless using --create-config)")
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = EZSortConfig()  # Default face age estimation
        
        print("🎯 EZ-Sort: Efficient Pairwise Annotation Tool")
        print("=" * 50)
        print(f"CSV file: {args.csv}")
        print(f"Image directory: {args.images}")
        print(f"Domain: {config.domain}")
        print(f"Buckets: {config.k_buckets}")
        print("=" * 50)
        
        # Load dataset
        print("\n📂 Loading dataset...")
        dataset = EZSortDataset(args.csv, args.images, args.image_col, args.label_col)
        
        # Initialize EZ-Sort
        print("\n🧠 Initializing EZ-Sort (running CLIP classification)...")
        annotator = EZSortAnnotator(dataset, config)
        
        # Run annotation session
        print(f"\n🚀 Starting annotation session...")
        results = simulate_annotation_session(
            annotator, 
            args.max_comparisons, 
            args.interactive
        )
        
        # Add final Elo scores to results
        results['final_elo_scores'] = annotator.elo_ratings.tolist()
        
        # Print summary
        print_summary(results)
        
        # Save results
        save_results(results, args.output, dataset)
        
        print(f"\n✅ EZ-Sort annotation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# install.py
"""
Easy installation script for EZ-Sort
Handles dependency installation and environment setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_banner():
    """Print installation banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    🎯 EZ-Sort Installer                    ║
    ║     Efficient Pairwise Annotation Tool (CIKM 2025)       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_gpu_support():
    """Check for GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU Support: {gpu_count} GPU(s) available ({gpu_name})")
            return True
        else:
            print("💻 CPU Only: No CUDA GPU detected (CLIP will run on CPU)")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed - will install with CPU support")
        return False


def run_command(command, description):
    """Run a command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"   Output: {e.output}")
        return False


def install_dependencies():
    """Install all dependencies"""
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip"):
        return False
    
    # Install PyTorch (with appropriate CUDA support)
    gpu_available = check_gpu_support()
    
    if platform.system() == "Darwin":  # macOS
        torch_command = f"{sys.executable} -m pip install torch torchvision"
    elif gpu_available:
        torch_command = f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_command = f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    
    if not run_command(torch_command, "Installing PyTorch"):
        return False
    
    # Install CLIP
    if not run_command(f"{sys.executable} -m pip install git+https://github.com/openai/CLIP.git", 
                      "Installing CLIP"):
        return False
    
    # Install other requirements
    requirements = [
        "streamlit>=1.25.0",
        "plotly>=5.15.0", 
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0"
    ]
    
    for req in requirements:
        if not run_command(f"{sys.executable} -m pip install '{req}'", f"Installing {req.split('>=')[0]}"):
            return False
    
    return True


def create_desktop_shortcuts():
    """Create desktop shortcuts for easy access"""
    
    current_dir = Path.cwd()
    
    if platform.system() == "Windows":
        # Windows .bat files
        
        # CLI shortcut
        cli_bat = current_dir / "ez_sort_cli.bat"
        with open(cli_bat, 'w') as f:
            f.write(f"""@echo off
cd /d "{current_dir}"
python ez_sort_cli.py %*
pause
""")
        
        # Web interface shortcut
        web_bat = current_dir / "ez_sort_web.bat"
        with open(web_bat, 'w') as f:
            f.write(f"""@echo off
cd /d "{current_dir}"
echo 🌐 Starting EZ-Sort Web Interface...
echo Open your browser to http://localhost:8501
python -m streamlit run ez_sort_web.py
pause
""")
        
        print("📂 Created Windows shortcuts:")
        print(f"   • {cli_bat}")
        print(f"   • {web_bat}")
    
    elif platform.system() in ["Linux", "Darwin"]:
        # Unix shell scripts
        
        # CLI shortcut
        cli_sh = current_dir / "ez_sort_cli.sh"
        with open(cli_sh, 'w') as f:
            f.write(f"""#!/bin/bash
cd "{current_dir}"
python ez_sort_cli.py "$@"
""")
        os.chmod(cli_sh, 0o755)
        
        # Web interface shortcut  
        web_sh = current_dir / "ez_sort_web.sh"
        with open(web_sh, 'w') as f:
            f.write(f"""#!/bin/bash
cd "{current_dir}"
echo "🌐 Starting EZ-Sort Web Interface..."
echo "Open your browser to http://localhost:8501"
python -m streamlit run ez_sort_web.py
""")
        os.chmod(web_sh, 0o755)
        
        print("📂 Created shell scripts:")
        print(f"   • {cli_sh}")
        print(f"   • {web_sh}")


def verify_installation():
    """Verify that installation was successful"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import clip
        import streamlit
        import pandas
        import numpy
        import scipy
        print("✅ All dependencies imported successfully")
        
        # Test CLIP loading
        print("🧠 Testing CLIP model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                 🎉 Installation Complete!                 ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🚀 Quick Start Options:
    
    1️⃣  Web Interface (Recommended for beginners):
       • Double-click: ez_sort_web.bat (Windows) or ez_sort_web.sh (Unix)
       • Or run: streamlit run ez_sort_web.py
       • Then open: http://localhost:8501
    
    2️⃣  Command Line:
       • Run demo: python demo.py
       • Interactive: python ez_sort_cli.py --csv data.csv --images img_dir/ --interactive
       • Simulation: python ez_sort_cli.py --csv data.csv --images img_dir/
    
    3️⃣  Examples:
       • Face age: python examples/face_age_example.py
       • Medical: python examples/medical_example.py
    
    📚 Documentation:
       • README.md - Complete usage guide
       • examples/ - Domain-specific examples
       • configs/ - Configuration templates
    
    🆘 Need Help?
       • GitHub Issues: https://github.com/your-username/ez-sort/issues
       • Email: your-email@university.edu
    """)


def main():
    """Main installation function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    print("\n📋 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")
    
    # Check GPU
    print("\n🖥️  Hardware Check:")
    check_gpu_support()
    
    # Install dependencies
    print("\n📦 Installing Dependencies:")
    if not install_dependencies():
        print("\n❌ Installation failed. Please check error messages above.")
        return False
    
    # Create shortcuts
    print("\n🔗 Creating Shortcuts:")
    create_desktop_shortcuts()
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Installation completed but verification failed.")
        print("   You may still be able to use EZ-Sort, but some features might not work.")
    
    # Print usage instructions
    print_usage_instructions()
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)


# quick_start.py
"""
Quick start script for EZ-Sort
Provides guided setup for first-time users
"""

import os
import sys
from pathlib import Path


def print_welcome():
    """Print welcome message"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                🎯 EZ-Sort Quick Start Guide                ║
    ║                                                           ║
    ║  This script will help you get started with EZ-Sort      ║
    ║  for efficient pairwise image annotation.                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def get_user_choice(prompt, options):
    """Get user choice from options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def setup_demo():
    """Set up and run demo"""
    print("\n🎮 Setting up demo...")
    
    demo_options = [
        "Face age estimation (30 synthetic faces)",
        "Medical image quality (40 synthetic images)", 
        "Custom demo with your parameters"
    ]
    
    choice = get_user_choice("Which demo would you like to run?", demo_options)
    
    if choice == 0:  # Face age demo
        print("\n🧑 Running face age estimation demo...")
        os.system("python examples/face_age_example.py")
        
    elif choice == 1:  # Medical demo
        print("\n🏥 Running medical image quality demo...")
        os.system("python examples/medical_example.py")
        
    elif choice == 2:  # Custom demo
        print("\n⚙️ Custom demo setup...")
        n_images = input("Number of images (default 30): ").strip() or "30"
        max_comps = input("Max comparisons (default 50): ").strip() or "50"
        
        print(f"\n🚀 Running custom demo with {n_images} images, {max_comps} comparisons...")
        os.system(f"python demo.py --n-images {n_images} --max-comparisons {max_comps}")


def setup_web_interface():
    """Launch web interface"""
    print("""
    🌐 Web Interface Setup
    
    The web interface provides an easy-to-use graphical interface
    for EZ-Sort annotation. It will open in your web browser.
    
    📋 You'll need:
    • CSV file with image paths and labels
    • Directory containing your images
    """)
    
    input("\nPress Enter to launch the web interface...")
    
    print("🚀 Launching web interface...")
    print("💡 Your browser will open to http://localhost:8501")
    print("💡 Press Ctrl+C in this terminal to stop the server")
    
    os.system("python -m streamlit run ez_sort_web.py")


def setup_cli():
    """Set up command line usage"""
    print("""
    💻 Command Line Interface Setup
    
    The CLI is perfect for batch processing and automation.
    """)
    
    has_data = input("\nDo you have your own dataset ready? (y/n): ").strip().lower()
    
    if has_data == 'y':
        csv_path = input("Enter path to your CSV file: ").strip()
        img_dir = input("Enter path to your image directory: ").strip()
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return
            
        if not os.path.exists(img_dir):
            print(f"❌ Image directory not found: {img_dir}")
            return
        
        interactive = input("Run interactive mode? (y/n): ").strip().lower() == 'y'
        max_comps = input("Maximum comparisons (default 100): ").strip() or "100"
        
        cmd = f"python ez_sort_cli.py --csv '{csv_path}' --images '{img_dir}' --max-comparisons {max_comps}"
        if interactive:
            cmd += " --interactive"
        
        print(f"\n🚀 Running: {cmd}")
        os.system(cmd)
        
    else:
        print("\n📚 CLI Examples:")
        print("1. Run with your data:")
        print("   python ez_sort_cli.py --csv data.csv --images img_dir/")
        print("\n2. Interactive mode:")
        print("   python ez_sort_cli.py --csv data.csv --images img_dir/ --interactive")
        print("\n3. Custom configuration:")
        print("   python ez_sort_cli.py --csv data.csv --images img_dir/ --config custom_config.json")
        print("\n💡 Create a demo dataset first with: python demo.py")


def setup_custom_domain():
    """Guide for setting up custom domain"""
    print("""
    🎨 Custom Domain Setup
    
    EZ-Sort can be adapted for any pairwise ranking task.
    Let's create a configuration for your domain.
    """)
    
    domain_name = input("Enter your domain name (e.g., 'product_quality', 'document_relevance'): ").strip()
    range_desc = input("Describe the ranking range (e.g., 'low to high quality'): ").strip()
    
    print(f"\n📝 Creating configuration for '{domain_name}' domain...")
    
    # Create custom config
    config_content = f'''{{
  "domain": "{domain_name}",
  "range_description": "{range_desc}",
  "k_buckets": 4,
  "hierarchical_prompts": {{
    "level_1": [
      "a {domain_name} image showing high quality characteristics",
      "a {domain_name} image showing low quality characteristics"
    ],
    "level_2": [
      "a {domain_name} image with excellent quality",
      "a {domain_name} image with good quality", 
      "a {domain_name} image with poor quality",
      "a {domain_name} image with very poor quality"
    ]
  }}
}}'''
    
    config_path = f"configs/{domain_name}_config.json"
    os.makedirs("configs", exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Configuration saved to: {config_path}")
    print("\n📝 Next steps:")
    print(f"1. Edit {config_path} to customize the prompts for your domain")
    print("2. Prepare your CSV file with image paths and labels")
    print("3. Run: python ez_sort_cli.py --csv your_data.csv --images your_images/ --config " + config_path)


def main():
    """Main quick start function"""
    print_welcome()
    
    # Check if EZ-Sort is properly installed
    try:
        import clip
        import streamlit
        print("✅ EZ-Sort is properly installed")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please run: python install.py")
        return
    
    main_options = [
        "🎮 Run Demo (recommended for first-time users)",
        "🌐 Launch Web Interface",
        "💻 Use Command Line Interface", 
        "🎨 Set up Custom Domain",
        "📚 View Documentation",
        "🚪 Exit"
    ]
    
    while True:
        choice = get_user_choice("What would you like to do?", main_options)
        
        if choice == 0:  # Demo
            setup_demo()
            
        elif choice == 1:  # Web interface
            setup_web_interface()
            
        elif choice == 2:  # CLI
            setup_cli()
            
        elif choice == 3:  # Custom domain
            setup_custom_domain()
            
        elif choice == 4:  # Documentation
            print("""
            📚 Documentation Links:
            
            • README.md - Complete user guide
            • examples/ - Usage examples for different domains
            • configs/ - Configuration templates
            • tests/ - Test suite and examples
            
            🌐 Online Resources:
            • GitHub: https://github.com/your-username/ez-sort
            • Paper: EZ-Sort (CIKM 2025)
            • Issues: https://github.com/your-username/ez-sort/issues
            """)
            
        elif choice == 5:  # Exit
            print("\n👋 Thank you for using EZ-Sort!")
            break
        
        input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()


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
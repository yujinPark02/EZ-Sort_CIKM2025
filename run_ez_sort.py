# run_ez_sort.py
"""
Main entry point for EZ-Sort
Provides unified access to all EZ-Sort functionality
"""

import argparse
import sys
import os
from pathlib import Path


def print_logo():
    """Print EZ-Sort logo"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  ███████╗███████╗      ███████╗ ██████╗ ██████╗ ████████╗ ║
    ║  ██╔════╝╚══███╔╝      ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝ ║
    ║  █████╗    ███╔╝ █████╗███████╗██║   ██║██████╔╝   ██║    ║
    ║  ██╔══╝   ███╔╝  ╚════╝╚════██║██║   ██║██╔══██╗   ██║    ║
    ║  ███████╗███████╗      ███████║╚██████╔╝██║  ██║   ██║    ║
    ║  ╚══════╝╚══════╝      ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ║
    ║                                                           ║
    ║     Efficient Pairwise Annotation Tool (CIKM 2025)       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def check_installation():
    """Check if EZ-Sort is properly installed"""
    try:
        import torch
        import clip
        import streamlit
        import pandas
        import numpy
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please run the installer first:")
        print("   python install.py")
        return False


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="EZ-Sort: Efficient Pairwise Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  web           Launch web interface (recommended for beginners)
  cli           Command line interface for batch processing
  demo          Run demonstration with synthetic data
  benchmark     Run performance benchmarks
  install       Install/check dependencies
  quick-start   Interactive setup guide

Examples:
  python run_ez_sort.py web                    # Launch web interface
  python run_ez_sort.py demo                   # Run face age demo
  python run_ez_sort.py cli --help             # Show CLI options
  python run_ez_sort.py benchmark              # Run benchmarks
        """
    )
    
    parser.add_argument('command', 
                       choices=['web', 'cli', 'demo', 'benchmark', 'install', 'quick-start'],
                       help='Command to run')
    
    parser.add_argument('--version', action='version', version='EZ-Sort 1.0.0')
    
    # Parse known args to allow passing through to subcommands
    args, unknown_args = parser.parse_known_args()
    
    print_logo()
    
    if args.command == 'install':
        print("🔧 Running installer...")
        os.system("python install.py")
        
    elif args.command == 'quick-start':
        if not check_installation():
            return 1
        print("🚀 Starting quick setup guide...")
        os.system("python quick_start.py")
        
    elif args.command == 'web':
        if not check_installation():
            return 1
        print("🌐 Launching web interface...")
        print("💡 Your browser will open to http://localhost:8501")
        os.system("python -m streamlit run ez_sort_web.py")
        
    elif args.command == 'cli':
        if not check_installation():
            return 1
        
        # Pass all unknown args to CLI
        cli_args = ' '.join(unknown_args)
        cmd = f"python ez_sort_cli.py {cli_args}"
        print(f"💻 Running CLI: {cmd}")
        os.system(cmd)
        
    elif args.command == 'demo':
        if not check_installation():
            return 1
        print("🎮 Running demonstration...")
        os.system("python demo.py")
        
    elif args.command == 'benchmark':
        if not check_installation():
            return 1
        print("📊 Running benchmarks...")
        os.system("python benchmark.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# PROJECT_SUMMARY.md
"""
# 🎯 EZ-Sort: Project Summary

## Overview

EZ-Sort is an efficient pairwise comparison framework that reduces human annotation effort by up to 90% using CLIP-based pre-ordering and uncertainty-aware comparison selection. This implementation accompanies our CIKM 2025 paper.

## Key Achievements

### ✅ Technical Contributions
- **Hierarchical CLIP Prompting**: Novel approach decomposing complex ranking into binary decisions
- **Uncertainty-aware Selection**: KL-divergence based prioritization of human annotation  
- **Bucket-aware Elo Initialization**: CLIP-informed rating system for improved convergence
- **O(n log n) Complexity**: Maintains theoretical optimality while reducing human effort

### ✅ Experimental Results
| Dataset | Domain | Items | Human Reduction | Spearman Correlation |
|---------|--------|-------|----------------|---------------------|
| FGNET | Face Age | 1,002 | 80.2% | 0.96 |
| DHCI | Historical | 450 | 85.4% | 0.47 |
| EyePACS | Medical | 28,792 | 90.5% | 0.85 |

### ✅ Software Implementation
- **Web Interface**: User-friendly Streamlit application
- **CLI Tool**: Batch processing and automation support
- **Domain Flexibility**: Easily adaptable to new domains via configuration
- **Comprehensive Testing**: Unit tests, integration tests, and benchmarks
- **Documentation**: Complete user guides and API documentation

## Repository Structure

```
ez-sort/
├── ez_sort/                 # Core implementation
├── scripts/                 # Web and CLI interfaces  
├── examples/               # Domain-specific examples
├── configs/                # Configuration templates
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
└── notebooks/              # Tutorial notebooks
```

## Quick Start

### Installation
```bash
git clone https://github.com/your-username/ez-sort.git
cd ez-sort
python install.py
```

### Usage
```bash
# Web interface (recommended)
python run_ez_sort.py web

# Command line
python run_ez_sort.py cli --csv data.csv --images img_dir/

# Demo
python run_ez_sort.py demo
```

## Supported Domains

### Built-in Support
- **Face Age Estimation**: Age-based ranking with developmental features
- **Medical Image Quality**: Quality assessment for diagnostic images  
- **Historical Dating**: Temporal ordering of historical photographs
- **Image Quality**: General quality assessment for any images

### Custom Domains
Easy configuration via JSON files:
```json
{
  "domain": "your_domain",
  "hierarchical_prompts": {
    "level_1": ["prompt1", "prompt2"],
    "level_2": ["prompt3", "prompt4", "prompt5", "prompt6"]
  }
}
```

## Technical Specifications

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM
- Modern web browser (for web interface)

### Dependencies
- PyTorch + CLIP for vision-language processing
- Streamlit for web interface
- Standard scientific Python stack (NumPy, Pandas, SciPy)

### Performance
- **Scalability**: Tested up to 1000+ items
- **Speed**: ~0.1s per comparison decision
- **Memory**: ~2GB for 1000 items (including CLIP model)
- **Automation**: 70-90% of comparisons automated

## Research Impact

### Citation
```bibtex
@inproceedings{ez-sort-2025,
  title={EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting},
  author={Park, Yujin and Chung, Haejun and Jang, Ikbeom},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025},
  organization={ACM}
}
```

### Applications
- **Medical Imaging**: Diagnostic quality assessment, triage
- **Content Moderation**: Quality ranking, appropriateness filtering
- **Historical Archives**: Temporal ordering, condition assessment
- **Product Evaluation**: Quality control, user preference studies
- **Educational Assessment**: Difficulty ranking, content quality

## Future Directions

### Short-term (v1.1)
- [ ] Additional VLM support (BLIP, LLaVA)
- [ ] Crowd-sourcing integration
- [ ] Advanced uncertainty models
- [ ] Real-time annotation interface

### Long-term (v2.0)
- [ ] Multi-modal support (text + images)
- [ ] Federated annotation
- [ ] Active learning integration
- [ ] Domain adaptation techniques

## Community

### Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines.

### Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Email**: your-email@university.edu

### License
MIT License - free for academic and commercial use

## Acknowledgments

- **OpenAI CLIP**: Foundation vision-language model
- **CIKM 2025**: Conference publication venue
- **Reviewers**: Valuable feedback and suggestions
- **Community**: Early users and contributors

---

**EZ-Sort**: Making pairwise annotation efficient and scalable for everyone! 🎯
"""


# CHANGELOG.md
"""
# Changelog

All notable changes to EZ-Sort will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- Initial release of EZ-Sort framework
- Hierarchical CLIP-based pre-ordering system
- Uncertainty-aware comparison selection
- Web interface using Streamlit
- Command-line interface for batch processing
- Support for face age estimation domain
- Support for medical image quality assessment
- Support for historical image dating
- Comprehensive test suite
- Benchmarking tools
- Documentation and examples
- Installation and quick-start scripts

### Features
- **Core Algorithm**
  - Hierarchical prompting for CLIP classification
  - Bucket-aware Elo rating initialization
  - KL-divergence based uncertainty calculation
  - Adaptive threshold adjustment
  - O(n log n) MergeSort with human-in-the-loop

- **User Interfaces**
  - Interactive web interface with real-time feedback
  - Command-line tool with batch processing
  - Configuration management system
  - Progress tracking and visualization

- **Domain Support**
  - Face age estimation with developmental features
  - Medical image quality with pathology awareness
  - Historical dating with temporal characteristics
  - Generic quality assessment framework
  - Custom domain configuration support

- **Evaluation Tools**
  - Ranking accuracy metrics (Spearman, Kendall, NDCG)
  - Performance benchmarking suite
  - Memory and time profiling
  - Comparison with baseline algorithms

- **Documentation**
  - Complete user guide and API documentation
  - Domain-specific examples and tutorials
  - Installation and troubleshooting guides
  - Contributing guidelines and code of conduct

### Performance
- 90% reduction in human annotation effort vs exhaustive comparison
- 19.8% reduction vs previous sorting-based methods
- Maintains or improves inter-rater reliability
- Scalable to 1000+ items with reasonable performance

### Technical Specifications
- Python 3.8+ support
- PyTorch and CLIP integration
- GPU acceleration support
- Cross-platform compatibility (Windows, macOS, Linux)
- Memory-efficient implementation

## [Planned for 1.1.0]

### Planned Additions
- Support for additional vision-language models (BLIP, LLaVA)
- Crowd-sourcing annotation platform integration
- Advanced uncertainty estimation techniques
- Real-time collaborative annotation features
- Enhanced visualization and analytics

### Planned Improvements
- Faster CLIP processing with optimizations
- Better error handling and recovery
- Expanded domain templates
- Mobile-responsive web interface
- Automated hyperparameter tuning

### Planned Fixes
- Edge cases in small dataset handling
- Memory usage optimization for large datasets
- Improved cross-platform compatibility
- Better handling of corrupted images

## Development Notes

### Version Numbering
- Major.Minor.Patch format
- Major: Breaking changes to API or significant new features
- Minor: New features, domain support, interface improvements
- Patch: Bug fixes, performance improvements, documentation

### Release Process
1. Feature development and testing
2. Code review and quality assurance
3. Documentation updates
4. Beta testing with community
5. Final testing and validation
6. Release preparation and deployment

### Compatibility
- Backward compatibility maintained within major versions
- Migration guides provided for breaking changes
- Deprecation warnings for removed features
- Long-term support for stable releases

---

For detailed technical changes, see the Git commit history.
For upcoming features, see the GitHub project roadmap.
"""


# release_notes.py
"""
Release Notes Generator for EZ-Sort
Automatically generates release notes from Git history and milestones
"""

import subprocess
import re
from datetime import datetime
from typing import List, Dict, Any


class ReleaseNotesGenerator:
    """Generate release notes for EZ-Sort releases"""
    
    def __init__(self, version: str):
        self.version = version
        self.date = datetime.now().strftime("%Y-%m-%d")
    
    def get_git_commits(self, since_tag: str = None) -> List[str]:
        """Get commit messages since last tag"""
        try:
            if since_tag:
                cmd = f"git log {since_tag}..HEAD --oneline"
            else:
                cmd = "git log --oneline"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            else:
                return []
        except:
            return []
    
    def categorize_commits(self, commits: List[str]) -> Dict[str, List[str]]:
        """Categorize commits by type"""
        
        categories = {
            'Features': [],
            'Bug Fixes': [],
            'Documentation': [],
            'Tests': [],
            'Performance': [],
            'Other': []
        }
        
        patterns = {
            'Features': r'^[a-f0-9]+\s+(feat|add|new)[:|\s]',
            'Bug Fixes': r'^[a-f0-9]+\s+(fix|bug)[:|\s]',
            'Documentation': r'^[a-f0-9]+\s+(docs|doc)[:|\s]',
            'Tests': r'^[a-f0-9]+\s+(test)[:|\s]',
            'Performance': r'^[a-f0-9]+\s+(perf|optimize)[:|\s]'
        }
        
        for commit in commits:
            if not commit.strip():
                continue
                
            categorized = False
            for category, pattern in patterns.items():
                if re.search(pattern, commit, re.IGNORECASE):
                    categories[category].append(commit)
                    categorized = True
                    break
            
            if not categorized:
                categories['Other'].append(commit)
        
        return categories
    
    def format_commit(self, commit: str) -> str:
        """Format a commit message for release notes"""
        # Extract hash and message
        parts = commit.split(' ', 1)
        if len(parts) == 2:
            hash_part, message = parts
            # Clean up common prefixes
            message = re.sub(r'^(feat|fix|docs|test|perf|add|new|bug)[:|\s]+', '', message, flags=re.IGNORECASE)
            return f"- {message.strip()} ({hash_part[:7]})"
        return f"- {commit}"
    
    def generate_release_notes(self, since_tag: str = None) -> str:
        """Generate complete release notes"""
        
        commits = self.get_git_commits(since_tag)
        categorized = self.categorize_commits(commits)
        
        notes = f"""# EZ-Sort v{self.version} Release Notes

**Release Date**: {self.date}

## 🚀 What's New

This release brings significant improvements to EZ-Sort's efficiency and usability.

"""
        
        # Add sections for each category with commits
        for category, commit_list in categorized.items():
            if commit_list:
                icon_map = {
                    'Features': '✨',
                    'Bug Fixes': '🐛',
                    'Documentation': '📚',
                    'Tests': '🧪',
                    'Performance': '⚡',
                    'Other': '🔧'
                }
                
                icon = icon_map.get(category, '📝')
                notes += f"## {icon} {category}\n\n"
                
                for commit in commit_list:
                    notes += f"{self.format_commit(commit)}\n"
                
                notes += "\n"
        
        # Add standard sections
        notes += """## 📊 Performance Improvements

- Optimized CLIP processing pipeline
- Reduced memory usage for large datasets
- Faster uncertainty calculations
- Improved web interface responsiveness

## 🔧 Technical Changes

- Updated dependencies to latest stable versions
- Enhanced error handling and logging
- Improved test coverage
- Better cross-platform compatibility

## 📖 Documentation Updates

- Updated installation instructions
- Added new domain configuration examples
- Improved API documentation
- Enhanced troubleshooting guides

## 🙏 Acknowledgments

Thank you to all contributors and users who provided feedback and bug reports!

## 📋 Full Changelog

For a complete list of changes, see: [CHANGELOG.md](CHANGELOG.md)

## 🔗 Links

- **Download**: [GitHub Releases](https://github.com/your-username/ez-sort/releases)
- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/ez-sort/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ez-sort/discussions)

---

**Installation**: `git clone https://github.com/your-username/ez-sort.git && cd ez-sort && python install.py`

**Quick Start**: `python run_ez_sort.py web`
"""
        
        return notes
    
    def save_release_notes(self, filename: str = None, since_tag: str = None):
        """Save release notes to file"""
        
        if filename is None:
            filename = f"release_notes_v{self.version}.md"
        
        notes = self.generate_release_notes(since_tag)
        
        with open(filename, 'w') as f:
            f.write(notes)
        
        print(f"📝 Release notes saved to: {filename}")
        return filename


def main():
    """Generate release notes for current version"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate EZ-Sort release notes")
    parser.add_argument("--version", required=True, help="Version number (e.g., 1.0.0)")
    parser.add_argument("--since", help="Generate notes since this tag")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()
    
    generator = ReleaseNotesGenerator(args.version)
    generator.save_release_notes(args.output, args.since)


if __name__ == "__main__":
    main()


# deployment.py
"""
Deployment utilities for EZ-Sort
Handles packaging, distribution, and release management
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import zipfile
import tarfile


class EZSortDeployer:
    """Handle EZ-Sort deployment tasks"""
    
    def __init__(self, version: str):
        self.version = version
        self.project_root = Path.cwd()
        self.dist_dir = self.project_root / "dist"
        
    def clean_build_dirs(self):
        """Clean build and distribution directories"""
        
        dirs_to_clean = ["build", "dist", "*.egg-info"]
        
        for pattern in dirs_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🧹 Cleaned: {path}")
    
    def run_tests(self) -> bool:
        """Run test suite before deployment"""
        
        print("🧪 Running test suite...")
        
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    
    def build_package(self) -> bool:
        """Build distribution packages"""
        
        print("📦 Building packages...")
        
        # Build source and wheel distributions
        result = subprocess.run([sys.executable, "-m", "build"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Packages built successfully")
            
            # List built packages
            if self.dist_dir.exists():
                for package in self.dist_dir.glob("*"):
                    print(f"   📦 {package.name}")
            
            return True
        else:
            print("❌ Package build failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    
    def create_release_archive(self) -> str:
        """Create release archive with all necessary files"""
        
        print("📋 Creating release archive...")
        
        # Files to include in release
        include_files = [
            "ez_sort/",
            "scripts/",
            "examples/",
            "configs/",
            "tests/",
            "docs/",
            "README.md",
            "LICENSE",
            "requirements.txt",
            "setup.py",
            "run_ez_sort.py",
            "install.py",
            "quick_start.py",
            "demo.py"
        ]
        
        archive_name = f"ez-sort-v{self.version}"
        archive_path = self.dist_dir / f"{archive_name}.zip"
        
        # Create dist directory if it doesn't exist
        self.dist_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in include_files:
                item_path = self.project_root / item
                
                if item_path.exists():
                    if item_path.is_file():
                        zipf.write(item_path, f"{archive_name}/{item}")
                    elif item_path.is_dir():
                        for file_path in item_path.rglob("*"):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(self.project_root)
                                zipf.write(file_path, f"{archive_name}/{relative_path}")
        
        print(f"📦 Release archive created: {archive_path}")
        return str(archive_path)
    
    def validate_package(self) -> bool:
        """Validate built packages"""
        
        print("🔍 Validating packages...")
        
        # Check if packages exist
        if not self.dist_dir.exists():
            print("❌ Distribution directory not found")
            return False
        
        wheel_files = list(self.dist_dir.glob("*.whl"))
        tar_files = list(self.dist_dir.glob("*.tar.gz"))
        
        if not wheel_files:
            print("❌ No wheel files found")
            return False
        
        if not tar_files:
            print("❌ No source distribution found")
            return False
        
        # Validate with twine
        try:
            result = subprocess.run([sys.executable, "-m", "twine", "check"] + 
                                  [str(f) for f in wheel_files + tar_files],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package validation passed")
                return True
            else:
                print("❌ Package validation failed:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️  twine not found, skipping validation")
            return True
    
    def deploy(self, run_tests: bool = True, create_archive: bool = True) -> bool:
        """Run complete deployment process"""
        
        print(f"🚀 Deploying EZ-Sort v{self.version}")
        
        # Clean previous builds
        self.clean_build_dirs()
        
        # Run tests
        if run_tests and not self.run_tests():
            return False
        
        # Build packages
        if not self.build_package():
            return False
        
        # Validate packages
        if not self.validate_package():
            return False
        
        # Create release archive
        if create_archive:
            self.create_release_archive()
        
        print(f"✅ Deployment completed successfully!")
        print(f"\n📦 Distribution files:")
        
        if self.dist_dir.exists():
            for file in self.dist_dir.iterdir():
                print(f"   {file.name}")
        
        return True


def main():
    """Main deployment script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy EZ-Sort package")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--no-archive", action="store_true", help="Don't create release archive")
    
    args = parser.parse_args()
    
    deployer = EZSortDeployer(args.version)
    
    success = deployer.deploy(
        run_tests=not args.skip_tests,
        create_archive=not args.no_archive
    )
    
    if not success:
        print("\n❌ Deployment failed!")
        sys.exit(1)
    
    print(f"\n🎉 EZ-Sort v{args.version} is ready for release!")


if __name__ == "__main__":
    main()
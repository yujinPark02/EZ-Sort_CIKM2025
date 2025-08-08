
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


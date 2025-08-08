# 🎯 EZ-Sort: Efficient Pairwise Annotation Tool

**EZ-Sort** is an efficient pairwise comparison framework that reduces human annotation effort by up to 90% using CLIP-based pre-ordering and uncertainty-aware comparison selection.

📄 **Paper**: *EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting* (CIKM 2025)
<img width="960" height="383" alt="image" src="https://github.com/user-attachments/assets/501f4688-4017-4c79-bc8c-ce301576c8a8" />

## ✨ Key Features

- **🚀 90% Reduction** in human annotation effort compared to exhaustive pairwise comparison
- **🧠 CLIP-based Pre-ordering** using hierarchical prompting for zero-shot semantic ranking
- **🎯 Uncertainty-aware Selection** to prioritize human effort on difficult comparisons
- **⚡ Efficient O(n log n)** complexity while maintaining MergeSort optimality
- **🌐 Web Interface** for easy interactive annotation
- **💻 CLI Support** for batch processing and automation
- **🔧 Customizable Prompts** for different domains (face age, medical, quality assessment, etc.)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for CLIP processing)

### Install from source

```bash
git clone https://github.com/your-username/ez-sort.git
cd ez-sort
pip install -r requirements.txt
```

### Dependencies
```bash
pip install torch torchvision clip-by-openai streamlit plotly pandas numpy pillow scipy
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

Create a CSV file with image paths and labels:

```csv
image_path,age
face_001.jpg,25
face_002.jpg,34
face_003.jpg,19
face_004.jpg,42
```

### 2. Web Interface (Recommended)

Launch the interactive web interface:

```bash
streamlit run ez_sort_web.py
```

Then:
1. Enter your CSV path and image directory
2. Configure domain and prompts
3. Click "Initialize EZ-Sort"
4. Start annotating!

### 3. Command Line Interface

#### Simulation Mode (using ground truth)
```bash
python ez_sort_cli.py \
    --csv data/faces.csv \
    --images data/face_images/ \
    --max-comparisons 100
```

#### Interactive Mode (human annotation)
```bash
python ez_sort_cli.py \
    --csv data/faces.csv \
    --images data/face_images/ \
    --interactive \
    --max-comparisons 50
```

#### Custom Configuration
```bash
python ez_sort_cli.py \
    --csv data/medical.csv \
    --images data/medical_images/ \
    --config config_medical.json \
    --max-comparisons 200
```

## ⚙️ Configuration

### Creating Custom Prompts

EZ-Sort uses hierarchical prompting for different domains. Create a configuration file:

```bash
python ez_sort_cli.py --create-config
```

This creates `config_face_age.json` with default face age estimation prompts.

### Custom Domain Example: Medical Image Quality

```json
{
  "domain": "medical",
  "range_description": "normal to severe pathology",
  "k_buckets": 3,
  "hierarchical_prompts": {
    "level_1": [
      "a medical image showing normal/healthy condition",
      "a medical image showing abnormal/pathological condition"
    ],
    "level_2": [
      "a medical image with no visible abnormalities",
      "a medical image with mild abnormalities",
      "a medical image with moderate abnormalities",
      "a medical image with severe abnormalities"
    ]
  }
}
```

### Custom Domain Example: Image Quality Assessment

```json
{
  "domain": "quality",
  "range_description": "low to high quality",
  "k_buckets": 4,
  "hierarchical_prompts": {
    "level_1": [
      "a high quality, clear and well-composed image",
      "a low quality, blurry or poorly composed image"
    ],
    "level_2": [
      "an excellent quality image with perfect clarity",
      "a good quality image with minor imperfections",
      "a poor quality image with noticeable issues",
      "a very poor quality image with major problems"
    ]
  }
}
```

### Custom Domain Example: Historical Dating

```json
{
  "domain": "historical",
  "range_description": "1900s to 2000s",
  "k_buckets": 4,
  "hierarchical_prompts": {
    "level_1": [
      "a historical photograph from early 20th century with vintage characteristics",
      "a modern photograph from late 20th century with contemporary features"
    ],
    "level_2": [
      "a photograph from 1900-1920s with sepia tones and formal poses",
      "a photograph from 1930-1950s with improved clarity and casual poses",
      "a photograph from 1960-1980s with color and modern composition",
      "a photograph from 1990s-2000s with digital quality and contemporary style"
    ]
  }
}
```

## 📊 Understanding the Results

### Output Files

After annotation, EZ-Sort generates:

- **`ez_sort_results.json`**: Complete results with all metadata
- **`comparisons.csv`**: History of all comparisons made
- **`final_ranking.csv`**: Final ranking of all items

### Key Metrics

- **Human Queries**: Comparisons requiring human input
- **Auto Decisions**: Comparisons resolved automatically
- **Automation Rate**: Percentage of automatic decisions
- **Uncertainty Scores**: Confidence in each comparison

### Efficiency Analysis

```python
import pandas as pd

# Load results
results_df = pd.read_csv("results/comparisons.csv")

# Calculate automation rate by uncertainty
results_df.groupby('type')['uncertainty'].mean()
```

## 🎛️ Advanced Configuration

### Algorithm Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `k_buckets` | Number of CLIP pre-ordering buckets | 5 | 3-7 |
| `theta_0` | Base uncertainty threshold | 0.15 | 0.05-0.3 |
| `alpha` | Budget sensitivity | 0.3 | 0.1-0.5 |
| `beta` | Accuracy sensitivity | 0.9 | 0.5-1.0 |
| `elo_k` | Elo learning rate | 32 | 16-64 |
| `gamma` | Cross-bucket priority multiplier | 1.2 | 1.0-2.0 |

### Domain-Specific Tuning

#### Face Age Estimation
- Use `k_buckets=5` for fine-grained age groups
- Higher `elo_k=32` for clear age differences
- Focus on facial development features in prompts

#### Medical Images
- Use `k_buckets=3` for pathology severity
- Lower `theta_0=0.1` for more human oversight
- Include anatomical landmarks in prompts

#### Quality Assessment
- Use `k_buckets=4` for quality levels
- Higher `gamma=1.5` for cross-quality comparisons
- Focus on technical image characteristics

## 🔬 Research & Citation

If you use EZ-Sort in your research, please cite our paper:

```bibtex
@inproceedings{ez-sort-2025,
  title={EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting},
  author={Park, Yujin and Chung, Haejun and Jang, Ikbeom},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025},
  organization={ACM}
}
```

### Key Contributions

1. **Hierarchical CLIP Prompting**: Novel approach to decompose complex ranking into binary decisions
2. **Uncertainty-aware Selection**: KL-divergence based prioritization of human annotation
3. **Bucket-aware Elo Initialization**: CLIP-informed rating system for improved convergence

### Experimental Results

| Dataset | Domain | Items | Human Reduction | Spearman Correlation |
|---------|--------|-------|----------------|---------------------|
| FGNET | Face Age | 1,002 | 80.2% | 0.96 |
| DHCI | Historical | 450 | 85.4% | 0.47 |
| EyePACS | Medical | 28,792 | 90.5% | 0.85 |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/ez-sort.git
cd ez-sort
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- **Issues**: [GitHub Issues](https://github.com/your-username/ez-sort/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ez-sort/discussions)
- **Email**: yujin1019a@hanyang.ac.kr

## 🌟 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the vision-language model
- [Streamlit](https://streamlit.io/) for the web interface framework
- CIKM 2025 reviewers for valuable feedback

## 📚 Related Work

- **Active Learning for Pairwise Comparisons**: [Jang et al., 2022]
- **CLIP-based Zero-shot Classification**: [Radford et al., 2021]
- **Human-in-the-Loop Machine Learning**: [Holzinger, 2016]

---

<div align="center">

**🎯 EZ-Sort: Making pairwise annotation efficient and scalable**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/Paper-CIKM%202025-green.svg)](https://your-paper-link.com)

</div>

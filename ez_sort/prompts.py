
# prompts.py
"""
Hierarchical prompt templates for different domains
"""

from typing import Dict, List
import json


def get_default_prompts(domain: str) -> Dict[str, List[str]]:
    """Get default hierarchical prompts for a domain"""
    
    prompts = {
        "face": {
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
        },
        
        "medical": {
            "level_1": [
                "a medical image showing normal, healthy anatomical structures",
                "a medical image showing abnormal or pathological findings"
            ],
            "level_2": [
                "a medical image with completely normal anatomy and no visible abnormalities",
                "a medical image with mild abnormalities or early pathological changes",
                "a medical image with moderate pathological findings requiring attention",
                "a medical image with severe abnormalities or advanced pathological conditions"
            ]
        },
        
        "quality": {
            "level_1": [
                "a high quality, sharp and well-composed photograph",
                "a low quality, blurry or poorly composed photograph"
            ],
            "level_2": [
                "an excellent quality photograph with perfect sharpness and professional composition",
                "a good quality photograph with minor technical imperfections but acceptable clarity",
                "a poor quality photograph with noticeable blur, noise, or composition issues",
                "a very poor quality photograph with major technical problems and severe quality issues"
            ]
        },
        
        "historical": {
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
        },
        
        "aesthetic": {
            "level_1": [
                "a visually appealing photograph with excellent aesthetic qualities",
                "a visually unappealing photograph with poor aesthetic qualities"
            ],
            "level_2": [
                "a stunning photograph with exceptional beauty and artistic composition",
                "an attractive photograph with good visual appeal and composition",
                "an average photograph with acceptable but unremarkable aesthetic qualities",
                "an unattractive photograph with poor visual appeal and composition"
            ]
        }
    }
    
    return prompts.get(domain, prompts["face"])


def create_custom_prompts(domain: str, 
                         categories: List[str],
                         descriptions: List[str]) -> Dict[str, List[str]]:
    """
    Create custom hierarchical prompts
    
    Args:
        domain: Domain name (e.g., "custom_domain")
        categories: List of category names
        descriptions: List of detailed descriptions for each category
    
    Returns:
        Dictionary with hierarchical prompts
    """
    
    if len(categories) != len(descriptions):
        raise ValueError("Categories and descriptions must have same length")
    
    # Create binary splits for hierarchical structure
    prompts = {}
    
    # Level 1: Binary split
    mid_point = len(categories) // 2
    prompts["level_1"] = [
        f"a {domain} image showing {', '.join(categories[:mid_point])}",
        f"a {domain} image showing {', '.join(categories[mid_point:])}"
    ]
    
    # Level 2: More detailed categories
    prompts["level_2"] = [f"a {domain} image with {desc}" for desc in descriptions]
    
    return prompts


def save_prompts_to_file(prompts: Dict[str, List[str]], 
                        filepath: str) -> None:
    """Save prompts to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"📄 Prompts saved to {filepath}")


def load_prompts_from_file(filepath: str) -> Dict[str, List[str]]:
    """Load prompts from JSON file"""
    with open(filepath, 'r') as f:
        prompts = json.load(f)
    return prompts


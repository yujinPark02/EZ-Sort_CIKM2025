"""
EZ-Sort Web Interface using Streamlit
Provides an interactive web interface for human annotation using the EZ-Sort framework.
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import time
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig
# (추가) 키보드 입력용
try:
    from streamlit_keypress import keypress  # pip install streamlit-keypress
    HAS_KEYPRESS = True
except Exception:
    import streamlit.components.v1 as components
    HAS_KEYPRESS = False
# Page config
st.set_page_config(
    page_title="EZ-Sort: Efficient Pairwise Annotation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.comparison-box {
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
}

.image-container {
    border: 3px solid #d1d5db;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem;
    text-align: center;
    transition: border-color 0.3s;
}

.image-container:hover {
    border-color: #3b82f6;
}

.selected {
    border-color: #10b981 !important;
    background-color: #ecfdf5;
}

.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.875rem;
    font-weight: 600;
}

.confidence-high { background-color: #dcfce7; color: #166534; }
.confidence-medium { background-color: #fef3c7; color: #92400e; }
.confidence-low { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


class SessionState:
    """Manage session state for the annotation interface"""
    
    def __init__(self):
        if 'annotator' not in st.session_state:
            st.session_state.annotator = None
        if 'current_pair' not in st.session_state:
            st.session_state.current_pair = None
        if 'comparison_queue' not in st.session_state:
            st.session_state.comparison_queue = []
        if 'results' not in st.session_state:
            st.session_state.results = {
                'comparisons': [],
                'human_queries': 0,
                'auto_decisions': 0,
                'ranking_history': []
            }
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0


def load_dataset(csv_path: str, image_dir: str, image_col: str, label_col: str) -> EZSortDataset:
    """Load dataset with error handling"""
    try:
        dataset = EZSortDataset(csv_path, image_dir, image_col, label_col)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def create_config_from_ui() -> EZSortConfig:
    """Create EZSortConfig from UI inputs"""
    
    # Domain selection
    domain = st.sidebar.selectbox(
        "Domain Type",
        ["face", "medical", "historical", "quality", "custom"],
        help="Select the type of data you're annotating"
    )
    
    # Bucket configuration
    k_buckets = st.sidebar.slider("Number of Buckets", 3, 7, 5)
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        theta_0 = st.slider("Base Uncertainty Threshold", 0.05, 0.3, 0.15)
        alpha = st.slider("Budget Sensitivity (α)", 0.1, 0.5, 0.3)
        beta = st.slider("Accuracy Sensitivity (β)", 0.5, 1.0, 0.9)
        elo_k = st.slider("Elo Learning Rate", 16, 64, 32)
    
    # Prompt configuration
    st.sidebar.subheader("Hierarchical Prompts")
    
    if domain == "face":
        prompts = {
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
        range_desc = "0-60+ years"
    elif domain == "medical":
        prompts = {
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
        range_desc = "normal to severe pathology"
        k_buckets = 3  # Fewer buckets for medical
    elif domain == "quality":
        prompts = {
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
        range_desc = "low to high quality"
    else:
        # Custom prompts
        st.sidebar.info("For custom domain, modify the prompts in the configuration file")
        prompts = None
        range_desc = "custom range"
    
    config = EZSortConfig(
        domain=domain,
        range_description=range_desc,
        hierarchical_prompts=prompts,
        k_buckets=k_buckets,
        theta_0=theta_0,
        alpha=alpha,
        beta=beta,
        elo_k=elo_k
    )
    
    return config


def display_comparison_interface(annotator: EZSortAnnotator, idx1: int, idx2: int):
    """Display the pairwise comparison interface"""
    
    st.markdown('<div class="sub-header">🤔 Which image ranks higher?</div>', unsafe_allow_html=True)
    
    # Calculate uncertainty
    uncertainty = annotator.calculate_uncertainty(idx1, idx2)
    
    # Uncertainty badge
    if uncertainty > 0.7:
        conf_class = "confidence-low"
        conf_text = "High Uncertainty"
    elif uncertainty > 0.4:
        conf_class = "confidence-medium"
        conf_text = "Medium Uncertainty"
    else:
        conf_class = "confidence-high"
        conf_text = "Low Uncertainty"
    
    st.markdown(f"""
    <div class="confidence-badge {conf_class}">
        {conf_text} (Score: {uncertainty:.3f})
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Display images
    with col1:
        try:
            img_path1 = annotator.dataset.get_image_path(idx1)
            img1 = Image.open(img_path1)
            st.image(img1, caption=f"Image A (Index: {idx1})", use_column_width=True)
            
            if st.button("🏆 Image A ranks higher", key="btn_a", type="primary"):
                return 1
                
        except Exception as e:
            st.error(f"Could not load image A: {e}")
    
    with col2:
        try:
            img_path2 = annotator.dataset.get_image_path(idx2)
            img2 = Image.open(img_path2)
            st.image(img2, caption=f"Image B (Index: {idx2})", use_column_width=True)
            
            if st.button("🏆 Image B ranks higher", key="btn_b", type="primary"):
                return 0
                
        except Exception as e:
            st.error(f"Could not load image B: {e}")
    
    # Skip button for difficult comparisons
    col_skip1, col_skip2, col_skip3 = st.columns([1, 1, 1])
    with col_skip2:
        if st.button("⏭️ Skip this comparison", key="btn_skip"):
            return "skip"
    
    return None


def display_progress_dashboard(results: Dict, annotator: EZSortAnnotator):
    """Display progress and metrics dashboard"""
    
    total_comparisons = results['human_queries'] + results['auto_decisions']
    automation_rate = results['auto_decisions'] / total_comparisons if total_comparisons > 0 else 0
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comparisons", total_comparisons)
    with col2:
        st.metric("Human Queries", results['human_queries'])
    with col3:
        st.metric("Auto Decisions", results['auto_decisions'])
    with col4:
        st.metric("Automation Rate", f"{automation_rate:.1%}")
    
    # Progress visualization
    if len(results['comparisons']) > 0:
        # Create comparison history chart
        df_hist = pd.DataFrame(results['comparisons'])
        
        fig = px.line(
            df_hist, 
            x='step', 
            y='uncertainty',
            color='type',
            title='Uncertainty vs. Comparison Type',
            labels={'uncertainty': 'Uncertainty Score', 'step': 'Comparison Step'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bucket distribution
        bucket_counts = np.bincount(annotator.bucket_assignments)
        
        fig_buckets = px.bar(
            x=list(range(len(bucket_counts))),
            y=bucket_counts,
            title='CLIP Pre-ordering: Bucket Distribution',
            labels={'x': 'Bucket ID', 'y': 'Number of Items'}
        )
        
        st.plotly_chart(fig_buckets, use_container_width=True)


def export_results(results: Dict, dataset: EZSortDataset):
    """Export annotation results"""
    
    st.subheader("📁 Export Results")
    
    export_format = st.selectbox("Export Format", ["CSV", "JSON"])
    
    if st.button("💾 Export Results"):
        if export_format == "CSV":
            # Create results DataFrame
            comparisons_df = pd.DataFrame(results['comparisons'])
            
            # Add final ranking
            final_ranking = results.get('final_ranking', [])
            ranking_df = pd.DataFrame({
                'image_index': final_ranking,
                'rank_position': range(len(final_ranking)),
                'image_path': [dataset.image_paths[i] for i in final_ranking] if final_ranking else []
            })
            
            # Create download buttons
            csv_comparisons = comparisons_df.to_csv(index=False)
            csv_ranking = ranking_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📊 Download Comparisons CSV",
                    csv_comparisons,
                    "ez_sort_comparisons.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "🏆 Download Ranking CSV", 
                    csv_ranking,
                    "ez_sort_ranking.csv",
                    "text/csv"
                )
        
        else:  # JSON
            json_data = json.dumps(results, indent=2)
            st.download_button(
                "📋 Download JSON Results",
                json_data,
                "ez_sort_results.json",
                "application/json"
            )


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    session = SessionState()
    
    # Header
    st.markdown('<div class="main-header">🎯 EZ-Sort: Efficient Pairwise Annotation Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **EZ-Sort** reduces human annotation effort by up to 90% using CLIP-based pre-ordering and uncertainty-aware comparison selection.
    
    📄 **Paper**: *EZ-Sort: Efficient Pairwise Comparison via Zero-Shot CLIP-Based Pre-Ordering and Human-in-the-Loop Sorting* (CIKM 2025)
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Dataset input
    st.sidebar.subheader("📂 Dataset")
    csv_path = st.sidebar.text_input("CSV File Path", placeholder="path/to/your/dataset.csv")
    image_dir = st.sidebar.text_input("Image Directory", placeholder="path/to/images/")
    image_col = st.sidebar.text_input("Image Column Name", value="image_path")
    label_col = st.sidebar.text_input("Label Column Name", value="label")
    
    # Load dataset button
    if st.sidebar.button("📁 Load Dataset"):
        if csv_path and image_dir:
            with st.spinner("Loading dataset..."):
                dataset = load_dataset(csv_path, image_dir, image_col, label_col)
                if dataset:
                    st.session_state.dataset = dataset
                    st.success(f"✅ Loaded {dataset.n_items} items")
        else:
            st.error("Please provide both CSV path and image directory")
    
    # Configuration
    config = create_config_from_ui()
    
    # Initialize EZ-Sort
    if 'dataset' in st.session_state and st.sidebar.button("🚀 Initialize EZ-Sort"):
        with st.spinner("Initializing EZ-Sort (running CLIP classification)..."):
            try:
                annotator = EZSortAnnotator(st.session_state.dataset, config)
                st.session_state.annotator = annotator
                
                # Initialize comparison queue (simplified)
                comparison_queue = []
                for i in range(min(100, annotator.n_items - 1)):  # Limit for demo
                    for j in range(i + 1, min(i + 10, annotator.n_items)):  # Limit pairs
                        comparison_queue.append((i, j))
                
                st.session_state.comparison_queue = comparison_queue
                st.session_state.current_step = 0
                
                st.success("✅ EZ-Sort initialized successfully!")
                
            except Exception as e:
                st.error(f"Error initializing EZ-Sort: {e}")
    
    # Main annotation interface
    if st.session_state.annotator is not None:
        
        # Progress dashboard
        with st.expander("📊 Progress Dashboard", expanded=True):
            display_progress_dashboard(st.session_state.results, st.session_state.annotator)
        
        # Current comparison
        if st.session_state.comparison_queue and st.session_state.current_step < len(st.session_state.comparison_queue):
            
            idx1, idx2 = st.session_state.comparison_queue[st.session_state.current_step]
            
            # Check if human query is needed
            should_query = st.session_state.annotator.should_query_human(
                idx1, idx2, st.session_state.current_step, len(st.session_state.comparison_queue)
            )
            
            if should_query:
                # Human comparison interface
                preference = display_comparison_interface(st.session_state.annotator, idx1, idx2)
                
                if preference is not None and preference != "skip":
                    # Process the comparison
                    st.session_state.annotator.update_elo(idx1, idx2, preference)
                    
                    # Record result
                    st.session_state.results['comparisons'].append({
                        'step': st.session_state.current_step,
                        'idx1': idx1,
                        'idx2': idx2,
                        'preference': preference,
                        'type': 'human',
                        'uncertainty': st.session_state.annotator.calculate_uncertainty(idx1, idx2)
                    })
                    
                    st.session_state.results['human_queries'] += 1
                    st.session_state.current_step += 1
                    
                    st.rerun()
                
                elif preference == "skip":
                    st.session_state.current_step += 1
                    st.rerun()
            
            else:
                # Auto comparison
                auto_preference = 1 if st.session_state.annotator.elo_ratings[idx1] > st.session_state.annotator.elo_ratings[idx2] else 0
                st.session_state.annotator.update_elo(idx1, idx2, auto_preference)
                
                # Record result
                st.session_state.results['comparisons'].append({
                    'step': st.session_state.current_step,
                    'idx1': idx1,
                    'idx2': idx2,
                    'preference': auto_preference,
                    'type': 'auto',
                    'uncertainty': st.session_state.annotator.calculate_uncertainty(idx1, idx2)
                })
                
                st.session_state.results['auto_decisions'] += 1
                st.session_state.current_step += 1
                
                st.info(f"🤖 Auto-decided: Image {idx1 if auto_preference else idx2} ranks higher (low uncertainty)")
                time.sleep(1)
                st.rerun()
        
        else:
            st.success("🎉 Annotation session completed!")
            
            # Get final ranking
            final_ranking = st.session_state.annotator.get_ranking()
            st.session_state.results['final_ranking'] = final_ranking
            
            # Export results
            export_results(st.session_state.results, st.session_state.dataset)
    
    else:
        st.info("👆 Please load a dataset and initialize EZ-Sort to begin annotation.")
        
        # Example usage
        with st.expander("📖 Example Usage"):
            st.markdown("""
            **1. Prepare your data:**
            - CSV file with image paths and labels
            - Image directory with actual images
            
            **2. Example CSV format:**
            ```csv
            image_path,age
            face_001.jpg,25
            face_002.jpg,34
            face_003.jpg,19
            ```
            
            **3. Configure prompts:**
            - Choose appropriate domain (face, medical, quality, etc.)
            - Adjust hierarchical prompts for your specific task
            
            **4. Start annotation:**
            - EZ-Sort will automatically pre-order using CLIP
            - You'll only be asked about uncertain comparisons
            - Export results when complete
            """)


if __name__ == "__main__":
    main()
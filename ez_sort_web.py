"""
EZ-Sort Web Interface using Streamlit (Full, fixed, English)
- Robust session_state initialization (fixes: AttributeError: st.session_state has no attribute "ui")
- Keyboard shortcuts: A (Image A), B (Image B), E (Equal), K (Skip) — no extra packages
- Face domain: Younger vs Older policy (affects UI wording and accuracy evaluation)
- Accuracy evaluation when labels exist (numeric); skipped otherwise
- Sample-size queue shaping and clear reporting
- Hierarchical prompts LEVEL 1..4 with detailed, descriptive adjectives across domains
"""

import json
import time
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# Your EZ-Sort backend
from ez_sort import EZSortDataset, EZSortAnnotator, EZSortConfig


# =========================
# Page Config & CSS
# =========================
st.set_page_config(
    page_title="EZ-Sort: Efficient Pairwise Annotation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_base_css():
    st.markdown(
        """
<style>
.main-header {
  font-size: 2.2rem; font-weight: 800; color: #1e3a8a;
  text-align: center; margin: 0.2rem 0 1.0rem;
}
.sub-header { font-size: 1.15rem; font-weight: 800; color: #374151; margin: 0.25rem 0 0.6rem; }
.metric-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0.75rem; border-radius: 10px; color: white; text-align: center; margin: 0.4rem 0;
}
.image-container {
  border: 2px solid #d1d5db; border-radius: 10px; padding: 0.25rem; margin: 0.25rem; text-align: center;
  transition: border-color 0.25s, background-color 0.25s;
}
.image-container:hover { border-color: #3b82f6; }
.confidence-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.82rem; font-weight: 800; }
.confidence-high { background-color: #dcfce7; color: #166534; }
.confidence-medium { background-color: #fef3c7; color: #92400e; }
.confidence-low { background-color: #fee2e2; color: #991b1b; }
kbd {
  display:inline-block; padding:0.15rem 0.4rem; font-size:0.85rem; line-height:1; color:#111827;
  background:#F3F4F6; border:1px solid #D1D5DB; border-bottom-width:2px; border-radius:6px; margin: 0 0.05rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

def inject_dynamic_image_css(max_height_px: int, width_pct: int):
    st.markdown(
        f"""
<style>
[data-testid="stImage"] img {{
  max-height: {max_height_px}px !important;
  height: auto !important;
  max-width: {width_pct}%;
  object-fit: contain !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )

inject_base_css()


# =========================
# Session-State (robust init)
# =========================
def ensure_session_state():
    """Ensure all required keys exist before any access."""
    ss = st.session_state
    ss.setdefault("annotator", None)
    ss.setdefault("dataset", None)
    ss.setdefault("comparison_queue", [])
    ss.setdefault("current_step", 0)
    ss.setdefault("subset_indices", [])
    ss.setdefault("results", {
        "comparisons": [],    # {step, idx1, idx2, preference, type, uncertainty, correct?}
        "human_queries": 0,
        "auto_decisions": 0,
        "n_eval": 0,
        "n_correct": 0,
    })
    ss.setdefault("ui", {
        "face_policy": "younger",   # "younger" | "older"
        "equal_tolerance": 1.0,     # label units (e.g., years)
    })

ensure_session_state()  # <<< this prevents the AttributeError you saw


# =========================
# Keyboard (A/B/E/K)
# =========================

def capture_hotkey() -> Optional[str]:
    """
    Version-agnostic global hotkey capture (A/B/E/K).
    - Uses a hidden text_input with a unique placeholder so JS can find it.
    - No components.html key=... (works on older Streamlit).
    - No extra packages. No Enter required.
    """
    placeholder = "__EZ_HOTKEY__"  # local, no global needed

    # Hidden buffer input that JS will write into
    st.text_input(
        "Hotkey buffer",
        value="",
        key="__hotkey__",
        max_chars=1,
        label_visibility="collapsed",
        placeholder=placeholder,
    )

    # Inject a small script that writes A/B/E/K into the hidden input and triggers a rerun
    components.html(
        f"""
        <script>
        (function() {{
          if (window.__ez_hotkey_inited__) return;
          window.__ez_hotkey_inited__ = true;

          const setVal = (k) => {{
            try {{
              const doc = window.parent?.document || document;
              const el = doc.querySelector('input[placeholder="{placeholder}"]');
              if (!el) return;
              el.value = k;
              el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }} catch (e) {{}}
          }};

          const onKey = (e) => {{
            const k = (e.key || '').toUpperCase();
            if (['A','B','E','K'].includes(k)) setVal(k);
          }};

          // Listen on both iframe and parent (when possible)
          window.addEventListener('keydown', onKey, true);
          try {{ window.parent.document.addEventListener('keydown', onKey, true); }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )

    v = st.session_state.get("__hotkey__", "")
    if v:
        st.session_state["__hotkey__"] = ""  # clear after read to avoid repeats
        return v.strip().upper()[:1]
    return None




# =========================
# Dataset & Config
# =========================
def load_dataset(csv_path: str, image_dir: str, image_col: str, label_col: str) -> Optional[EZSortDataset]:
    try:
        return EZSortDataset(csv_path, image_dir, image_col, label_col)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def _prompts_face(levels: list[int]) -> dict[str, list[str]]:
    """Face-age hierarchical prompts with detailed visual descriptors (Levels 1–4)."""
    P = {}
    if 1 in levels:
        P["level1"] = [
            # 0: Child (0-17)
            "a photograph of a child or teenager's face with rounded features, smoother skin with no wrinkles, "
            "larger eyes relative to face, smaller nose and jaw, and no signs of aging like nasolabial folds. "
            "This face has the soft, unweathered appearance of youth.",
            # 1: Adult (18+)
            "a photograph of an adult face with fully defined features, mature facial structure including prominent cheekbones, "
            "developed jawline, deeper set eyes, potential facial lines, and visible signs of aging like nasolabial folds or crow's feet."
        ]
    if 2 in levels:
        P["level2"] = [
            # 0_0: Very Young Child (0-8)
            "a photograph of a very young child (0-8 years) with distinctly childlike features: prominently rounded cheeks, "
            "disproportionately large forehead, very large eyes relative to face size, tiny nose and undeveloped chin, baby-like facial proportions, "
            "and the characteristic facial softness of early childhood.",
            # 0_1: Older Child/Teen (9-17)
            "a photograph of an older child or teenager (9-17 years) with adolescent features: more defined facial structure starting to emerge, "
            "less facial roundness than young children, more proportional forehead, developing jaw and cheekbones, possible acne, "
            "and the transitional features of puberty and adolescence.",
            # 1_0: Young Adult (18-35)
            "a photograph of a young adult (18-35 years) with fully developed yet youthful features: defined facial structure, "
            "clear skin elasticity, sharper jawline, mature proportions, minimal to no wrinkles, "
            "and the characteristic facial vitality of early adulthood without significant aging signs.",
            # 1_1: Mature Adult (36+)
            "a photograph of a mature adult (36+ years) with established aging signs: decreased skin elasticity, visible facial lines especially "
            "around eyes and mouth, more pronounced nasolabial folds, potential jowls or neck sagging, possibly graying hair, "
            "and the characteristic facial maturity of middle-age and beyond."
        ]
    if 3 in levels:
        P["level3"] = [
            # Infants and Toddlers (0-2)
            "a photograph of an infant or toddler (0-2 years) with extremely rounded cheeks, disproportionately large head and forehead, tiny features, "
            "undefined chin, barely visible neck, and the distinctive facial plumpness of babies.",
            # Young Children (3-8)
            "a photograph of a young child (3-8 years) with round cheeks but more defined than infants, large eyes relative to face, small nose, "
            "developing chin structure, primary teeth, and the characteristic innocent expression of early childhood.",
            # Preteens (9-12)
            "a photograph of a preteen (9-12 years) with faces beginning to lose childlike roundness, more defined features but still soft, mixed dentition, "
            "no significant facial hair in boys, pre-pubescent features, and the characteristic appearance of late childhood.",
            # Teenagers (13-17)
            "a photograph of a teenager (13-17 years) with emerging adult facial structure, defined jawline starting to appear, possible acne, "
            "growing nose proportions, emerging facial hair in males, developing secondary sexual characteristics, and the characteristic appearance of adolescence.",
            # Very Young Adults (18-25)
            "a photograph of a very young adult (18-25 years) with fully developed yet extremely fresh features, taut skin with high elasticity, "
            "defined facial structure but with youthful softness, minimal to no facial lines, bright complexion, and the characteristic vigorous appearance of early adulthood.",
            # Young Adults (26-35)
            "a photograph of a young adult (26-35 years) with completely mature facial features, defined contours, possible earliest fine lines around eyes when smiling, "
            "full facial definition, optimal skin elasticity beginning to slightly reduce, and the characteristic appearance of established adulthood.",
            # Middle-aged Adults (36-50)
            "a photograph of a middle-aged adult (36-50 years) with visible aging progression, established facial lines especially around eyes and nasolabial area, "
            "decreasing skin plumpness, possible early sagging, potential graying at temples, and the characteristic appearance of middle age.",
            # Seniors (51+)
            "a photograph of a senior adult (51+ years) with pronounced aging features, established wrinkles and facial folds, noticeably decreased skin elasticity, "
            "visible volume loss in cheeks, more defined nasolabial folds, often gray or white hair, and the characteristic appearance of advanced age."
        ]
    if 4 in levels:
        P["level4"] = [
            "a photograph of a newborn to 1-year-old infant with extremely round face, disproportionately large head, fontanelle (soft spot) may be visible, "
            "undefined facial features, minimal neck definition, and unfocused gaze.",
            "a photograph of a 1-2 year old toddler with very round cheeks, small but slightly more defined features than newborns, "
            "beginning facial expressions, first teeth may be visible, and beginning neck definition.",
            "a photograph of a 3-5 year old preschooler with distinctly childlike proportions, rounded cheeks but more definition than toddlers, "
            "primary teeth, more controlled facial expressions, defined eyes relative to face, and characteristic preschool appearance.",
            "a photograph of a 6-8 year old child with less facial roundness than younger children, early mixed dentition, more proportional features, "
            "defined nose and ears, and the characteristic appearance of early school age.",
            "a photograph of a 9-11 year old with faces beginning transition from child to adolescent, mixed dentition well established, "
            "more defined chin, losing the last of baby fat in cheeks, and pre-pubertal appearance.",
            "a photograph of a 12-14 year old early teenager with early pubertal changes, possible acne, growth in nose and jaw, "
            "early facial hair in males, and early adolescent features.",
            "a photograph of a 15-16 year old teenager with substantial pubertal development, near-adult facial proportions, "
            "established acne in many cases, significant growth in facial features, facial hair in males, and mid-adolescent appearance.",
            "a photograph of a 17-18 year old late teenager with nearly complete adolescent development, almost adult facial structure, "
            "defined jawline especially in males, possible residual acne, and late adolescent appearance approaching adulthood.",
            "a photograph of a 19-21 year old young adult with newly mature features, complete facial development, fresh complexion, "
            "optimal elasticity, and vibrant early adulthood appearance.",
            "a photograph of a 22-25 year old adult with fully established facial structure, optimal skin condition, defined contours, "
            "no visible aging signs, and the characteristic appearance of young adulthood.",
            "a photograph of a 26-30 year old adult with mature features, earliest very fine lines around eyes when smiling, "
            "excellent skin condition but first subtle elasticity changes.",
            "a photograph of a 31-35 year old adult with earliest fine lines at rest, slight reduction in facial plumpness, and pre-middle-age appearance.",
            "a photograph of a 36-42 year old adult with first definitive aging signs, established fine lines around eyes and possible forehead, "
            "slight skin laxity changes, and early middle age appearance.",
            "a photograph of a 43-50 year old adult with clear aging progression, deeper nasolabial folds, possible marionette lines, "
            "decreased skin elasticity, possible early jowl formation, and middle age appearance.",
            "a photograph of a 51-60 year old adult with pronounced wrinkles, visible volume loss, neck laxity, often gray hair, "
            "and the characteristic appearance of mature middle age.",
            "a photograph of a 61+ year old senior with significant wrinkles, clear volume loss, defined jowls, neck laxity and banding, "
            "thinning skin with age spots, and advanced age appearance."
        ]
    return P


def _prompts_medical(levels: List[int]) -> Dict[str, List[str]]:
    """Adjective-rich but domain-agnostic imaging cues (edges, margin, density, texture)."""
    P = {}
    if 1 in levels:
        P["level_1"] = [
            "a medical image with normal anatomy, uniform background, sharp boundaries, and no focal abnormality",
            "a medical image with abnormal findings, disrupted anatomy, focal opacity or defect, and atypical texture",
        ]
    if 2 in levels:
        P["level_2"] = [
            "no visible abnormality: crisp structures, symmetric appearance, homogeneous intensity",
            "mild abnormality: small focal change, well-circumscribed margins, subtle density shift",
            "moderate abnormality: multifocal changes, partially ill-defined margins, heterogeneous texture",
            "severe abnormality: diffuse involvement, spiculated or invasive margins, marked intensity distortion",
        ]
    if 3 in levels:
        P["level_3"] = [
            "normal variant patterns: anatomical landmarks intact, smooth contours, low noise",
            "localized lesion: round/oval, smooth or lobulated edge, mild contrast uptake",
            "regional disease: segmental involvement, reticular or granular texture, mass effect",
            "diffuse severe disease: widespread opacities, architectural distortion, edema-like signal",
        ]
    if 4 in levels:
        P["level_4"] = [
            "artifact or noise only: ringing, motion streaks, but preserved anatomical planes",
            "tiny lesion: <5mm, high contrast to background, sharp rim, no surrounding edema",
            "complex lesion: mixed signal core, rim enhancement, peri-lesional edema or halo",
            "infiltrative pattern: ill-defined spicules, vessel/ductal tracking, mass effect on adjacent tissue",
        ]
    return P

def _prompts_quality(levels: List[int]) -> Dict[str, List[str]]:
    """Perceptual image quality with clear photographic adjectives."""
    P = {}
    if 1 in levels:
        P["level_1"] = [
            "a high-quality photo: tack-sharp focus, low noise, balanced exposure, accurate color, clean composition",
            "a low-quality photo: motion blur or defocus, high noise, clipped highlights or crushed shadows, poor framing",
        ]
    if 2 in levels:
        P["level_2"] = [
            "excellent: crisp micro-contrast, natural white balance, no artifacts, strong subject isolation",
            "good: slight softness or mild ISO noise, minor clipping, acceptable framing",
            "poor: noticeable blur or grain, color cast, uneven exposure, distracting background",
            "very poor: heavy blur, severe noise/banding, harsh clipping, chaotic composition",
        ]
    if 3 in levels:
        P["level_3"] = [
            "razor-sharp edges, rich tonal gradation, smooth bokeh, no halos",
            "minor softness, fine luminance noise, small haloing, stable tones",
            "clear blur trails or smear, blotchy chroma noise, haloing/ghosting",
            "severe smear, streak noise, posterization, extreme vignetting",
        ]
    if 4 in levels:
        P["level_4"] = [
            "studio-grade clarity: high MTF, precise WB, no compression artifacts, controlled highlights",
            "near-studio: light noise, gentle roll-off, mild compression, clean edges",
            "consumer: visible sharpening halos, WB drift, aggressive noise reduction artifacts",
            "degraded: macro-blocking, color bleeding, blown highlights, muddy shadows",
        ]
    return P

def _prompts_historical(levels: List[int]) -> Dict[str, List[str]]:
    """Visual period cues for dating historical color images (generic, not dataset-specific)."""
    P = {}
    if 1 in levels:
        P["level_1"] = [
            "a historical photo with earlier aesthetic cues: muted palette, coarse grain, simple styling",
            "a historical photo with later aesthetic cues: more saturated palette, finer grain, modern styling",
        ]
    if 2 in levels:
        P["level_2"] = [
            "earlier era: subdued colors, thick film grain, conservative fashion, rounded car silhouettes",
            "mid era: moderate saturation, cleaner grain, transitional fashion, boxier industrial design",
            "later era: vivid colors, fine grain, contemporary fashion, angular product design",
        ]
    if 3 in levels:
        P["level_3"] = [
            "pre-50s aesthetic: sepia tint, heavy grain, soft focus, vintage signage and typography",
            "60s-70s aesthetic: warm color cast, moderate grain, bold patterns, chrome trim on vehicles",
            "80s-90s aesthetic: neutral cast, finer grain, synthetic fabrics, squared electronics",
            "post-90s aesthetic: crisp edges, high saturation, modern fonts, compact electronics",
        ]
    if 4 in levels:
        P["level_4"] = [
            "very early: orthographic look, uneven exposure, soft corners, hand-painted signboards",
            "early-mid: muted dyes, noticeable grain clumps, film halation around highlights",
            "mid-late: improved dye stability, cleaner edges, period-specific hairstyles and eyewear",
            "late: strong anti-halation, high resolving power, contemporary silhouettes and materials",
        ]
    return P

def build_hierarchical_prompts(domain: str, levels: List[int]) -> Optional[Dict[str, List[str]]]:
    if domain == "face":
        return _prompts_face(levels)
    if domain == "medical":
        return _prompts_medical(levels)
    if domain == "quality":
        return _prompts_quality(levels)
    if domain == "historical":
        return _prompts_historical(levels)
    # custom: user may edit later
    return None


def create_config_from_ui() -> EZSortConfig:
    """
    Builds EZSortConfig from sidebar.
    NOTE: we defensively (re)initialize st.session_state.ui here too,
    so this function is safe to call even if ensure_session_state() was removed.
    """
    st.session_state.setdefault("ui", {"face_policy": "younger", "equal_tolerance": 1.0})

    domain = st.sidebar.selectbox("Domain Type", ["face", "medical", "historical", "quality", "custom"])
    k_buckets = st.sidebar.slider("Number of Buckets (for CLIP pre-ordering)", 3, 7, 5)

    with st.sidebar.expander("Advanced Parameters"):
        theta_0 = st.slider("Base Uncertainty Threshold", 0.05, 0.30, 0.15)
        alpha   = st.slider("Budget Sensitivity (α)", 0.10, 0.50, 0.30)
        beta    = st.slider("Accuracy Sensitivity (β)", 0.50, 1.00, 0.90)
        elo_k   = st.slider("Elo Learning Rate (K)", 8, 64, 32)

    # Policy and prompts up to Level 4 with rich adjectives
    levels = [1, 2, 3, 4]
    prompts = build_hierarchical_prompts(domain, levels)
    range_desc = {
        "face": "0–80+ years (visual aging cues only)",
        "medical": "normal → severe pathology (margins, density, texture)",
        "quality": "very poor → studio-grade photographic quality",
        "historical": "early → late period visual cues",
        "custom": "custom range",
    }[domain]

    if domain == "face":
        st.session_state.ui["face_policy"] = st.sidebar.radio(
            "Face Comparison Policy", ["younger", "older"],
            help="Decide whether the preferred image is the one that looks younger or older.",
            index=0 if st.session_state.ui.get("face_policy","younger") == "younger" else 1
        )

    if domain == "medical":
        k_buckets = min(k_buckets, 4)

    return EZSortConfig(
        domain=domain,
        range_description=range_desc,
        hierarchical_prompts=prompts,
        k_buckets=k_buckets,
        theta_0=theta_0,
        alpha=alpha,
        beta=beta,
        elo_k=elo_k,
    )


# =========================
# UI Helpers / Dashboard
# =========================
def display_progress_dashboard(results: Dict[str, Any], annotator: EZSortAnnotator):
    total = results["human_queries"] + results["auto_decisions"]
    automation_rate = results["auto_decisions"] / total if total > 0 else 0.0
    accuracy = (results["n_correct"] / results["n_eval"]) if results["n_eval"] > 0 else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Comparisons", total)
    c2.metric("Human Queries", results["human_queries"])
    c3.metric("Auto Decisions", results["auto_decisions"])
    c4.metric("Evaluated (labels present)", results["n_eval"])
    c5.metric("Accuracy", "-" if accuracy is None else f"{accuracy*100:.1f}%")

    if results["comparisons"]:
        df_hist = pd.DataFrame(results["comparisons"])
        if "uncertainty" in df_hist:
            fig = px.line(
                df_hist, x="step", y="uncertainty", color="type",
                title="Uncertainty over time (human vs auto)",
                labels={"uncertainty": "uncertainty (0=confident, 1=uncertain)", "step": "step"},
            )
            st.plotly_chart(fig, use_container_width=True)

        if hasattr(annotator, "bucket_assignments") and annotator.bucket_assignments is not None:
            counts = np.bincount(annotator.bucket_assignments)
            fig_b = px.bar(
                x=list(range(len(counts))), y=counts,
                title="CLIP Pre-ordering: Bucket Distribution",
                labels={"x": "bucket id", "y": "count"},
            )
            st.plotly_chart(fig_b, use_container_width=True)

    with st.expander("Uncertainty vs Accuracy (what they mean)"):
        st.markdown(
            "- **Uncertainty**: how hard the model thinks a pair is (near 0.5 predicted win-probability).  \n"
            "- **Accuracy**: how often your human decision matches the ground-truth label rule (when labels exist).  \n"
            "They correlate (hard pairs are often wrong) but **they are not the same metric**."
        )
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Return 1..n ranks with average handling for ties (Spearman)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # tie-average
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks

def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")

def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b with tie correction."""
    n = len(x)
    C = D = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        prod = dx * dy
        C += int((prod > 0).sum())
        D += int((prod < 0).sum())
    # ties
    Tx = sum(c * (c - 1) // 2 for c in Counter(x).values() if c > 1)
    Ty = sum(c * (c - 1) // 2 for c in Counter(y).values() if c > 1)
    denom = np.sqrt((C + D + Tx) * (C + D + Ty))
    return float((C - D) / denom) if denom > 0 else 0.0

def _icc2_1(matrix: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, absolute agreement, single rater/measurement.
    Expect matrix shape (n_subjects, 2) — here: [pred_score, true_score].
    For scale compatibility we z-score each column first.
    """
    X = np.asarray(matrix, float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 2:
        return float("nan")

    # z-score columns to bring scales in line
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    if np.any(~np.isfinite(X)):
        return float("nan")

    n, k = X.shape  # k=2
    mean_t = X.mean(axis=1, keepdims=True)
    mean_r = X.mean(axis=0, keepdims=True)
    grand = X.mean()

    SSR = k * np.sum((mean_t - grand) ** 2)                 # rows/targets
    SSC = n * np.sum((mean_r - grand) ** 2)                 # columns/raters
    SSE = np.sum((X - mean_t - mean_r + grand) ** 2)        # residual

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1) if k > 1 else 0.0
    MSE = SSE / ((n - 1) * (k - 1)) if k > 1 else 0.0

    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    return float((MSR - MSE) / denom) if denom > 0 else float("nan")

def compute_final_metrics_for_subset(
    final_ranking: List[int],
    subset_indices: List[int],
    dataset: EZSortDataset,
    domain: str,
    face_policy: str
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Build a per-item table (subset only) and compute Spearman, Kendall tau-b, ICC(2,1)
    between the final ordering and labels (if numeric). ICC is computed on z-scored
    [pred_score, true_score] so scales are comparable.
    """
    subset = set(subset_indices)
    ranked_subset = [i for i in final_ranking if i in subset]
    n = len(ranked_subset)

    # Assemble table
    rows = []
    for pos, idx in enumerate(ranked_subset):
        label_val = get_label_value(dataset, idx)
        rows.append({
            "image_index": idx,
            "rank_position": pos,  # 0=best/top
            "image_path": dataset.image_paths[idx],
            "label": label_val
        })
    table = pd.DataFrame(rows)

    # If labels missing or not enough items — metrics become NaN
    if n < 2 or table["label"].isna().any():
        metrics = {"n": n, "spearman_rho": float("nan"), "kendall_tau_b": float("nan"), "icc": float("nan")}
        return metrics, table

    # Build scores: higher score should mean "better/top"
    # predicted score: invert rank so top gets higher score
    pred_score = -(table["rank_position"].to_numpy().astype(float))

    # true score: depends on policy for face; generic domains default to ascending=better
    labels = table["label"].to_numpy().astype(float)
    if domain == "face":
        # younger policy ⇒ lower age is better ⇒ true score = -age
        true_score = -labels if face_policy == "younger" else labels
    else:
        # If your task defines "higher label = better", keep labels.
        # Otherwise adapt here; for now we assume higher label = better.
        true_score = labels

    # Spearman
    rho = _pearsonr(_rankdata(pred_score), _rankdata(true_score))

    # Kendall tau-b
    tau_b = _kendall_tau_b(pred_score, true_score)

    # ICC(2,1) on z-scored columns
    icc_val = _icc2_1(np.column_stack([pred_score, true_score]))

    metrics = {"n": n, "spearman_rho": rho, "kendall_tau_b": tau_b, "icc": icc_val}
    return metrics, table


def export_results(
    results: Dict[str, Any],
    dataset: EZSortDataset,
    cfg: EZSortConfig,
    subset_indices: List[int]
):
    """Export only the sampled subset to CSV; also compute and show final ranking metrics."""
    st.subheader("Export Results")

    # Final ranking (filter to subset only)
    final_ranking = results.get("final_ranking", [])
    if not final_ranking:
        st.info("Final ranking is empty. Nothing to export yet.")
        return

    face_policy = st.session_state.ui.get("face_policy", "younger")
    metrics, subset_table = compute_final_metrics_for_subset(
        final_ranking=final_ranking,
        subset_indices=subset_indices,
        dataset=dataset,
        domain=cfg.domain,
        face_policy=face_policy,
    )

    # Show metrics inline
    st.markdown("**Final ranking accuracy (subset only):**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (subset)", metrics["n"])
    c2.metric("Spearman ρ", "-" if np.isnan(metrics["spearman_rho"]) else f"{metrics['spearman_rho']:.3f}")
    c3.metric("Kendall τ-b", "-" if np.isnan(metrics["kendall_tau_b"]) else f"{metrics['kendall_tau_b']:.3f}")
    c4.metric("ICC(2,1)", "-" if np.isnan(metrics["icc"]) else f"{metrics['icc']:.3f}")

    # Export format and buttons
    fmt = st.selectbox("Export Format", ["CSV", "JSON"])

    # Comparisons are already subset-only if you built the queue from the subset,
    # but we still filter defensively.
    comp_df = pd.DataFrame(results["comparisons"])
    if not comp_df.empty:
        subset_set = set(subset_indices)
        comp_df = comp_df[comp_df["idx1"].isin(subset_set) & comp_df["idx2"].isin(subset_set)].reset_index(drop=True)

    if st.button("Save"):
        if fmt == "CSV":
            # 1) Pairwise comparisons (subset only)
            st.download_button(
                "Download Comparisons CSV (subset pairs)",
                comp_df.to_csv(index=False),
                "ez_sort_comparisons_subset.csv",
                "text/csv",
            )
            # 2) Final ranking table (subset only, exactly N rows if N sample)
            st.download_button(
                "Download Ranking CSV (subset only)",
                subset_table.to_csv(index=False),
                "ez_sort_ranking_subset.csv",
                "text/csv",
            )
        else:
            payload = {
                "metrics": metrics,
                "comparisons": comp_df.to_dict(orient="records"),
                "ranking_subset": subset_table.to_dict(orient="records"),
            }
            st.download_button(
                "Download Results JSON (subset + metrics)",
                json.dumps(payload, indent=2),
                "ez_sort_results_subset.json",
                "application/json",
            )



# =========================
# Label & Accuracy Helpers
# =========================
def _to_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}: return None
        return float(s)
    except Exception:
        return None

def get_label_value(dataset: EZSortDataset, idx: int) -> Optional[float]:
    # Try multiple access patterns, depending on your EZSortDataset implementation
    try:
        return _to_float(dataset.labels[idx])  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return _to_float(dataset.df[dataset.label_col].iloc[idx])  # type: ignore[attr-defined]
    except Exception:
        pass
    return None

def eval_correctness_face(l1: float, l2: float, pref: float, policy: str, tol: float) -> Optional[bool]:
    """
    pref: 1.0 (A wins), 0.0 (B wins), 0.5 (Equal)
    policy: 'younger' or 'older' — defines which side should be preferred
    tol: absolute difference treated as equal (within tolerance)
    """
    if l1 is None or l2 is None:
        return None
    d = l1 - l2  # positive if A is older
    if abs(d) <= tol:
        return True if pref == 0.5 else False
    if policy == "younger":
        if pref == 1.0:   # chose A
            return l1 < l2
        elif pref == 0.0: # chose B
            return l2 < l1
        else:
            return False
    else:  # older
        if pref == 1.0:
            return l1 > l2
        elif pref == 0.0:
            return l2 > l1
        else:
            return False

def update_accuracy_metrics(idx1: int, idx2: int, pref: float, cfg: EZSortConfig):
    l1 = get_label_value(st.session_state.dataset, idx1)
    l2 = get_label_value(st.session_state.dataset, idx2)
    if l1 is None or l2 is None:
        return  # skip evaluation if either label is missing

    tol = float(st.session_state.ui.get("equal_tolerance", 1.0))
    if cfg.domain == "face":
        correct = eval_correctness_face(l1, l2, pref, st.session_state.ui.get("face_policy", "younger"), tol)
    else:
        correct = None  # For other domains we need a domain-specific rule; skipping by default.

    if correct is not None:
        st.session_state.results["n_eval"] += 1
        st.session_state.results["n_correct"] += int(bool(correct))


# =========================
# Elo update helper (draw-safe)
# =========================
def apply_elo_update_safe(annotator: EZSortAnnotator, i: int, j: int, preference: float):
    """
    preference: 1.0 (i wins), 0.0 (j wins), 0.5 (draw)
    Try native draw support; otherwise emulate draw conservatively.
    """
    try:
        annotator.update_elo(i, j, preference)
        return
    except TypeError:
        pass
    if preference == 0.5:
        try:
            old_k = getattr(annotator, "elo_k", None)
            if old_k is not None:
                annotator.elo_k = max(1, int(old_k // 4))
            annotator.update_elo(i, j, 1)  # i wins
            annotator.update_elo(j, i, 1)  # j wins back
        except Exception:
            pass
        finally:
            if old_k is not None:
                annotator.elo_k = old_k
    else:
        annotator.update_elo(i, j, int(preference))


# =========================
# Comparison UI
# =========================
def display_comparison_interface(annotator: EZSortAnnotator, idx1: int, idx2: int, cfg: EZSortConfig) -> Optional[float]:
    """Pairwise comparison UI with explicit input-based hotkeys:
       Type A (Image A), D (Image B), s (Equal) in the EZ_HOTKEY box and press Enter to apply.
    """
    # Wording based on face policy
    if cfg.domain == "face":
        policy = st.session_state.ui.get("face_policy", "younger")
        question = "Which face looks **younger**?" if policy == "younger" else "Which face looks **older**?"
    else:
        question = "Which image ranks **higher**?"

    st.markdown(f'<div class="sub-header">🤔 {question}</div>', unsafe_allow_html=True)

    # Uncertainty ribbon
    uncertainty = annotator.calculate_uncertainty(idx1, idx2)
    badge = ("confidence-low", "High Uncertainty") if uncertainty > 0.7 else \
            ("confidence-medium", "Medium Uncertainty") if uncertainty > 0.4 else \
            ("confidence-high", "Low Uncertainty")
    st.markdown(
        f'<div class="confidence-badge {badge[0]}">{badge[1]} (Score: {uncertainty:.3f})</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    pref: Optional[float] = None

    with col1:
        try:
            img1 = Image.open(annotator.dataset.get_image_path(idx1))
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img1, caption=f"Image A (Index: {idx1})", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("🏆 Image A (A)", key=f"btn_a_{idx1}_{idx2}", use_container_width=True):
                pref = 1.0
        except Exception as e:
            st.error(f"Could not load image A: {e}")

    with col2:
        try:
            img2 = Image.open(annotator.dataset.get_image_path(idx2))
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img2, caption=f"Image B (Index: {idx2})", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("🏆 Image B (D)", key=f"btn_b_{idx1}_{idx2}", use_container_width=True):
                pref = 0.0
        except Exception as e:
            st.error(f"Could not load image B: {e}")

    c_eq, c_skip = st.columns(2)
    with c_eq:
        if st.button("⚖️ Equal (s)", key=f"btn_equal_{idx1}_{idx2}", use_container_width=True):
            pref = 0.5
    with c_skip:
        if st.button("⏭️ Skip", key=f"btn_skip_{idx1}_{idx2}", use_container_width=True):
            pref = "skip"

    # --- Explicit input-based hotkey (type then press Enter) ---
    hk = st.text_input(
        "EZ_HOTKEY (type A for Image A, D for Image B, s for Equal, press Enter)",
        key="EZ_HOTKEY",
        max_chars=1,
    )
    # When Enter is pressed, Streamlit reruns and hk contains the typed char
    if hk:
        ch = hk.strip()
        if ch in ("A", "a"):
            pref = 1.0
        elif ch in ("D", "d"):
            pref = 0.0
        elif ch == "s":  # lowercase 's' as requested
            pref = 0.5
        # clear after use so it won't repeat on next rerun
        st.session_state["EZ_HOTKEY"] = ""

    return pref

# =========================
# Main App
# =========================
def main():
    st.markdown('<div class="main-header">🎯 EZ-Sort: Efficient Pairwise Annotation Tool</div>', unsafe_allow_html=True)
    st.markdown(
        "CLIP-based zero-shot **pre-ordering** + bucket-aware **Elo** + **uncertainty-aware** human routing. "
        "Keyboard: <kbd>A</kbd>/<kbd>B</kbd>/<kbd>E</kbd>/<kbd>K</kbd>."
    )

    # Sidebar: dataset
    st.sidebar.title("Configuration")
    st.sidebar.subheader("Dataset")
    csv_path  = st.sidebar.text_input("CSV File Path", placeholder="path/to/dataset.csv")
    image_dir = st.sidebar.text_input("Image Directory", placeholder="path/to/images/")
    image_col = st.sidebar.text_input("Image Column Name", value="image_path")
    label_col = st.sidebar.text_input("Label Column Name (optional)", value="label")

    if st.sidebar.button("Load Dataset"):
        if csv_path and image_dir:
            with st.spinner("Loading dataset..."):
                dataset = load_dataset(csv_path, image_dir, image_col, label_col)
                if dataset:
                    st.session_state.dataset = dataset
                    st.success(f"Loaded {dataset.n_items} items.")
        else:
            st.error("Please provide both CSV path and image directory.")

    # Sidebar: image display
    st.sidebar.subheader("Image Display")
    max_h = st.sidebar.slider("Max Image Height (px)", 200, 900, 440, step=10)
    width_pct = st.sidebar.slider("Column Width Percentage (%)", 50, 100, 100, step=5)
    inject_dynamic_image_css(max_h, width_pct)

    # Config (prompts/params)
    config = create_config_from_ui()

    # Sidebar: sampling / queue shaping
    st.sidebar.subheader("Sampling / Queue")
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)
    sample_size = st.sidebar.number_input("Sample Size (items to annotate)", min_value=10, value=100, step=10)
    neighbor_k = st.sidebar.slider("Neighbors per Anchor (pairs)", 2, 50, 10, step=1)
    human_budget = st.sidebar.number_input("Max Human Queries (budget)", min_value=1, value=100000, step=1)

    # Sidebar: evaluation
    st.sidebar.subheader("Evaluation")
    st.session_state.ui["equal_tolerance"] = float(
        st.sidebar.number_input("Equal tolerance (label units)", min_value=0.0, value=1.0, step=0.5)
    )

    # Initialize
    if "dataset" in st.session_state and st.session_state.dataset is not None and st.sidebar.button("Initialize EZ-Sort"):
        with st.spinner("Initializing (CLIP pre-ordering, buckets, Elo, uncertainty)…"):
            try:
                np.random.seed(seed)
                annotator = EZSortAnnotator(st.session_state.dataset, config)
                st.session_state.annotator = annotator

                # Sample subset of items
                n_items = annotator.n_items
                chosen = np.random.choice(n_items, size=min(sample_size, n_items), replace=False)
                chosen = sorted(chosen.tolist())
                st.session_state.subset_indices = chosen

                # Build comparison queue within the chosen subset
                queue = []
                for ix, i in enumerate(chosen):
                    for j in chosen[ix + 1 : ix + 1 + neighbor_k]:
                        queue.append((i, j))

                # Shuffle for variety (deterministic with seed)
                rng = np.random.default_rng(seed)
                rng.shuffle(queue)

                st.session_state.comparison_queue = queue
                st.session_state.current_step = 0
                st.session_state.results = {
                    "comparisons": [],
                    "human_queries": 0,
                    "auto_decisions": 0,
                    "n_eval": 0,
                    "n_correct": 0,
                }

                # Clear summary: proves sample_size is actually used
                unique_items_in_queue = sorted(set([p[0] for p in queue] + [p[1] for p in queue]))
                st.success(
                    f"Initialized.\n\n"
                    f"- Selected sample items: **{len(chosen)}** / {n_items}\n"
                    f"- Unique items appearing in queue: **{len(unique_items_in_queue)}**\n"
                    f"- Total pairs in queue: **{len(queue)}**  (≈ sample_size × neighbors per anchor)"
                )

            except Exception as e:
                st.error(f"Error initializing EZ-Sort: {e}")

    # Main loop
    if st.session_state.annotator is not None:
        with st.expander("Progress Dashboard", expanded=True):
            display_progress_dashboard(st.session_state.results, st.session_state.annotator)

        # Budget stop
        if st.session_state.results["human_queries"] >= human_budget:
            st.warning("Human query budget reached. Finalizing…")
            st.session_state.current_step = len(st.session_state.comparison_queue)

        if st.session_state.comparison_queue and st.session_state.current_step < len(st.session_state.comparison_queue):
            idx1, idx2 = st.session_state.comparison_queue[st.session_state.current_step]
            should_query = st.session_state.annotator.should_query_human(
                idx1, idx2, st.session_state.current_step, len(st.session_state.comparison_queue)
            )

            if should_query and (st.session_state.results["human_queries"] < human_budget):
                pref = display_comparison_interface(st.session_state.annotator, idx1, idx2, config)

                if pref is not None and pref != "skip":
                    apply_elo_update_safe(st.session_state.annotator, idx1, idx2, float(pref))
                    update_accuracy_metrics(idx1, idx2, float(pref), config)

                    st.session_state.results["comparisons"].append({
                        "step": st.session_state.current_step,
                        "idx1": idx1,
                        "idx2": idx2,
                        "preference": pref,
                        "type": "human",
                        "uncertainty": st.session_state.annotator.calculate_uncertainty(idx1, idx2),
                    })
                    st.session_state.results["human_queries"] += 1
                    st.session_state.current_step += 1
                    st.rerun()

                elif pref == "skip":
                    st.session_state.current_step += 1
                    st.rerun()

            else:
                auto_pref = 1.0 if st.session_state.annotator.elo_ratings[idx1] > st.session_state.annotator.elo_ratings[idx2] else 0.0
                apply_elo_update_safe(st.session_state.annotator, idx1, idx2, auto_pref)

                st.session_state.results["comparisons"].append({
                    "step": st.session_state.current_step,
                    "idx1": idx1,
                    "idx2": idx2,
                    "preference": auto_pref,
                    "type": "auto",
                    "uncertainty": st.session_state.annotator.calculate_uncertainty(idx1, idx2),
                })
                st.session_state.results["auto_decisions"] += 1
                st.session_state.current_step += 1
                st.info(f"Auto-decided: {'A' if auto_pref==1.0 else 'B'} (low uncertainty)")
                time.sleep(0.35)
                st.rerun()

        else:
            st.success("Annotation session completed!")
            try:
                final_ranking = st.session_state.annotator.get_ranking()
                st.session_state.results["final_ranking"] = final_ranking
            except Exception:
                st.session_state.results["final_ranking"] = []
        
            # Export (subset-only + metrics)
            export_results(
                st.session_state.results,
                st.session_state.dataset,
                config,
                st.session_state.subset_indices,
            )

    else:
        st.info("Load a dataset and initialize EZ-Sort to begin.")
        with st.expander("Quick Guide"):
            st.markdown(
                "- Put image paths in CSV (column name configurable). Optional numeric labels enable accuracy.\n"
                "- Choose domain and review Level 1–4 prompts (they drive CLIP zero-shot pre-ordering).\n"
                "- Use **Sample Size** and **Neighbors per Anchor** to shape the queue size.\n"
                "- Keyboard: <kbd>A</kbd>/<kbd>B</kbd>/<kbd>E</kbd>/<kbd>K</kbd>."
            )


if __name__ == "__main__":
    main()

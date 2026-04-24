"""
streamlit_app.py  (Home / Dashboard)
--------------------------------------
Run from within the app/ directory:
    cd app
    streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_models, load_vectorizer, load_results, load_transfer_model,
    load_raw_dataset, MODEL_META, ACCURACY_DISPLAY, NOTEBOOK_META,
    TL_SST2_ACCURACY, TL_MODEL_NAME,
)

# Page config
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS — dark glassmorphism theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0d1b2a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.3);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Gradient hero header ── */
.hero-banner {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #06B6D4 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.4);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(255,255,255,0.05) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: white;
    margin: 0.75rem 0.25rem 0 0;
    font-weight: 500;
    backdrop-filter: blur(10px);
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(124, 58, 237, 0.6);
    background: rgba(124, 58, 237, 0.1);
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.2);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Model cards ── */
.model-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(79, 70, 229, 0.3);
    border-radius: 18px;
    padding: 1.75rem;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(12px);
    height: 100%;
}
.model-card:hover {
    border-color: rgba(124, 58, 237, 0.7);
    background: rgba(124, 58, 237, 0.08);
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(124, 58, 237, 0.25);
}
.model-icon {
    font-size: 2.4rem;
    margin-bottom: 0.6rem;
}
.model-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.model-type {
    font-size: 0.8rem;
    color: #a78bfa;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.model-desc {
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.accuracy-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7C3AED, #4F46E5);
    border-radius: 50px;
    padding: 5px 16px;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
}
.speed-tag {
    display: inline-block;
    background: rgba(6, 182, 212, 0.15);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 50px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: #67e8f9;
    margin-left: 0.5rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(124,58,237,0.5), transparent);
    margin-left: 1rem;
}

/* ── Notebook pills ── */
.notebook-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.notebook-card:hover {
    border-color: rgba(6, 182, 212, 0.5);
    background: rgba(6, 182, 212, 0.06);
    transform: translateX(4px);
}
.nb-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 0.3rem;
}
.nb-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.nb-tag {
    display: inline-block;
    background: rgba(124, 58, 237, 0.2);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 50px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #c4b5fd;
    margin: 2px 2px 0 0;
}

/* ── Dividers ── */
hr { border-color: rgba(124, 58, 237, 0.2) !important; }

/* ── Plotly backgrounds ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: rgba(124, 58, 237, 0.8) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 10px !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(124, 58, 237, 0.4) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
.stSuccess {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
.stWarning {
    background: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-radius: 10px !important;
}
.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 10px !important;
}
.stInfo {
    background: rgba(6, 182, 212, 0.08) !important;
    border: 1px solid rgba(6, 182, 212, 0.25) !important;
    border-radius: 10px !important;
    color: #67e8f9 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.8rem;'>🛍️</div>
        <div style='font-size:1rem; font-weight:700; color:#a78bfa; margin-top:0.3rem;'>
            Amazon Sentiment
        </div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:0.2rem;'>
            ML Research Project
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    st.page_link("streamlit_app.py", label="🏠 Home / Dashboard", )
    st.page_link("pages/1_Single_Analysis.py", label="✍️ Single Review Analysis")
    st.page_link("pages/2_Batch_Analysis.py", label="📂 Batch File Analysis")
    st.page_link("pages/3_Model_Comparison.py", label="📊 Model Comparison")
    st.page_link("pages/4_About.py", label="📖 About & Notebooks")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#475569; line-height:1.7;'>
        <b style='color:#7C3AED;'>Classical Models</b><br>
        📈 Logistic Regression<br>
        🌲 Random Forest<br>
        📊 Naive Bayes<br><br>
        <b style='color:#7C3AED;'>Transfer Learning</b><br>
        🤗 DistilBERT (SST-2)<br><br>
        <b style='color:#7C3AED;'>Vectorizer</b><br>
        CountVectorizer · WordPiece
    </div>
    """, unsafe_allow_html=True)

# Load resources
with st.spinner("Loading models..."):
    models = load_models()
    vectorizer = load_vectorizer()
    results = load_results()
    raw_df = load_raw_dataset()

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛍️ Amazon Review<br>Sentiment Analysis</div>
    <div class="hero-subtitle">
        A comprehensive ML research project exploring classical &amp; deep learning
        approaches to binary sentiment classification on Amazon product reviews.
    </div>
    <div style='margin-top:1rem;'>
        <span class="hero-badge">🔬 4 ML Models</span>
        <span class="hero-badge">🤗 Transfer Learning (DistilBERT)</span>
        <span class="hero-badge">📓 4 Research Notebooks</span>
        <span class="hero-badge">🗂️ 1,000 Labelled Reviews</span>
        <span class="hero-badge">⚡ Real-time Inference</span>
        <span class="hero-badge">📂 Batch Upload Support</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Metrics
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

total_reviews = len(raw_df) if not raw_df.empty else 1000
pos_count = int((raw_df["Sentiment"] == "Positive").sum()) if not raw_df.empty else 500
neg_count = int((raw_df["Sentiment"] == "Negative").sum()) if not raw_df.empty else 500
pos_pct = round(pos_count / total_reviews * 100, 1)
neg_pct = round(neg_count / total_reviews * 100, 1)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, str(total_reviews), "Total Reviews"),
    (m2, str(pos_count), "Positive Reviews"),
    (m3, str(neg_count), "Negative Reviews"),
    (m4, f"{pos_pct}%", "Positive Rate"),
    (m5, "4", "Models Deployed"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Class Distribution Chart
st.markdown("")
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Negative"],
        values=[pos_count, neg_count],
        hole=0.55,
        marker=dict(
            colors=["#10B981", "#EF4444"],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textfont=dict(family="Inter", size=13, color="white"),
        hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
    ))
    fig_pie.add_annotation(
        text=f"<b>Dataset</b><br>Balance",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#94a3b8", family="Inter"),
        xanchor="center", yanchor="middle",
    )
    fig_pie.update_layout(
        title=dict(text="Class Distribution", font=dict(color="#e2e8f0", size=15, family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        legend=dict(font=dict(color="#e2e8f0", family="Inter"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=20, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_chart2:
    model_names = list(ACCURACY_DISPLAY.keys())
    acc_vals = [round(results.get(ACCURACY_DISPLAY[m], 0) * 100, 1) for m in model_names]

    fig_bar = go.Figure()
    colors = ["#7C3AED", "#4F46E5", "#06B6D4"]
    for i, (name, acc, color) in enumerate(zip(model_names, acc_vals, colors)):
        fig_bar.add_trace(go.Bar(
            x=[name], y=[acc],
            marker=dict(
                color=color,
                line=dict(width=0),
            ),
            name=name,
            text=[f"{acc}%"],
            textposition="outside",
            textfont=dict(color="white", size=12, family="Inter"),
            hovertemplate=f"<b>{name}</b><br>Accuracy: {acc}%<extra></extra>",
            width=0.45,
        ))
    fig_bar.update_layout(
        title=dict(text="Model Accuracy (Test Set)", font=dict(color="#e2e8f0", size=15, family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        showlegend=False,
        xaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(family="Inter")),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            range=[0, 100], color="#94a3b8",
            ticksuffix="%", tickfont=dict(family="Inter"),
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=300,
        bargap=0.4,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Model Cards
st.markdown('<div class="section-header">🤖 Deployed Models</div>', unsafe_allow_html=True)

classical_names = [n for n in MODEL_META if not MODEL_META[n].get("is_transfer")]
model_cols = st.columns(3)
for idx, model_name in enumerate(classical_names):
    meta    = MODEL_META[model_name]
    acc_key = ACCURACY_DISPLAY.get(model_name, "")
    acc     = results.get(acc_key, 0)
    acc_display = f"{round(acc * 100, 1)}%" if acc else "N/A"
    with model_cols[idx]:
        st.markdown(f"""
        <div class="model-card">
            <div class="model-icon">{meta['icon']}</div>
            <div class="model-name">{model_name}</div>
            <div class="model-type">{meta['type']}</div>
            <div class="model-desc">{meta['description']}</div>
            <div>
                <span class="accuracy-badge">🎯 {acc_display} Accuracy</span>
                <span class="speed-tag">{meta['speed']}</span>
            </div>
            <div style='margin-top:0.75rem; font-size:0.78rem; color:#64748b;'>
                Vectorizer: <span style='color:#a78bfa;'>{meta['vectorizer']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
tl_meta = MODEL_META.get(TL_MODEL_NAME, {})
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(251,191,36,0.06) 0%, rgba(245,158,11,0.08) 50%, rgba(234,88,12,0.06) 100%);
    border: 1px solid rgba(251,191,36,0.35);
    border-left: 4px solid #F59E0B;
    border-radius: 18px;
    padding: 1.75rem 2rem;
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
">
    <div style='font-size:3rem; flex-shrink:0;'>{tl_meta.get('icon','🤗')}</div>
    <div style='flex:1;'>
        <div style='display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.4rem;'>
            <span style='font-size:1.1rem; font-weight:800; color:#e2e8f0;'>{TL_MODEL_NAME}</span>
            <span style='background:linear-gradient(135deg,#F59E0B,#EF4444); border-radius:50px; padding:3px 12px;
                font-size:0.72rem; font-weight:700; color:white;'>🚀 Transfer Learning</span>
            <span style='background:rgba(251,191,36,0.15); border:1px solid rgba(251,191,36,0.35); border-radius:50px;
                padding:3px 12px; font-size:0.72rem; color:#fbbf24;'>SST-2 Benchmark: {round(TL_SST2_ACCURACY*100,1)}%</span>
        </div>
        <div style='font-size:0.78rem; font-weight:600; color:#F59E0B; text-transform:uppercase;
            letter-spacing:0.08em; margin-bottom:0.6rem;'>{tl_meta.get('type','')}</div>
        <div style='font-size:0.875rem; color:#94a3b8; line-height:1.65; margin-bottom:1rem;'>{tl_meta.get('description','')}</div>
        <div style='display:flex; gap:0.5rem; flex-wrap:wrap;'>
            <span style='background:rgba(251,191,36,0.15); border:1px solid rgba(251,191,36,0.3); border-radius:50px;
                padding:4px 14px; font-size:0.78rem; color:#fbbf24;'>🤗 HuggingFace Transformers</span>
            <span style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:50px;
                padding:4px 14px; font-size:0.78rem; color:#94a3b8;'>Tokenizer: {tl_meta.get('vectorizer','WordPiece')}</span>
            <span style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:50px;
                padding:4px 14px; font-size:0.78rem; color:#94a3b8;'>Speed: {tl_meta.get('speed','')}</span>
            <span style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:50px;
                padding:4px 14px; font-size:0.78rem; color:#94a3b8;'>Downloads ~250 MB on first use</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Research Notebooks overview
st.markdown('<div class="section-header">📓 Research Notebooks</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem;'>"
    "Four notebooks explore progressively more complex NLP approaches — from classical ML "
    "to deep learning and transfer learning architectures.</p>",
    unsafe_allow_html=True,
)

nb_cols = st.columns(2)
for i, nb in enumerate(NOTEBOOK_META):
    tags_html = "".join(f'<span class="nb-tag">{t}</span>' for t in nb["tags"])
    with nb_cols[i % 2]:
        st.markdown(f"""
        <div class="notebook-card">
            <div class="nb-title">📓 {nb['title']}</div>
            <div class="nb-desc">{nb['desc']}</div>
            {tags_html}
        </div>
        """, unsafe_allow_html=True)

# Quick CTA
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 1rem 0;'>
    <div style='font-size:1.2rem; font-weight:700; color:#e2e8f0; margin-bottom:0.4rem;'>
        Ready to analyse a review?
    </div>
    <div style='font-size:0.9rem; color:#64748b;'>
        Use the sidebar navigation to run single review analysis or upload a file for batch processing.
    </div>
</div>
""", unsafe_allow_html=True)

cta_c1, cta_c2, cta_c3 = st.columns([1,1,1])
with cta_c1:
    st.page_link("pages/1_Single_Analysis.py", label="✍️ Single Analysis →")
with cta_c2:
    st.page_link("pages/2_Batch_Analysis.py", label="📂 Batch Upload →")
with cta_c3:
    st.page_link("pages/3_Model_Comparison.py", label="📊 Model Charts →")

st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem; font-size:0.75rem; color:#334155;'>
    Amazon Review Sentiment Analysis · Built with Streamlit & scikit-learn · 
    <span style='color:#7C3AED;'>Research Project</span>
</div>
""", unsafe_allow_html=True)
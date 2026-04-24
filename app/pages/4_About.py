# Project info, tech stack, and research notebook overview

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from utils import NOTEBOOK_META, MODEL_META, ACCURACY_DISPLAY, load_results

st.set_page_config(
    page_title="About · Amazon Sentiment",
    page_icon="📖", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:linear-gradient(135deg,#0f0c29 0%,#1a1040 40%,#0d1b2a 100%);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,12,41,0.95)!important;border-right:1px solid rgba(124,58,237,0.3);}
#MainMenu,footer,header{visibility:hidden;}
hr{border-color:rgba(124,58,237,0.2)!important;}
.page-header{background:linear-gradient(135deg,#1e1b4b 0%,#2d1b69 50%,#1e3a5f 100%);border:1px solid rgba(167,139,250,0.4);border-radius:18px;padding:2rem 2.5rem;margin-bottom:2rem;box-shadow:0 8px 32px rgba(124,58,237,0.2);}
.page-title{font-size:2rem;font-weight:800;color:#e2e8f0;margin:0 0 0.3rem 0;}
.page-sub{font-size:0.95rem;color:#94a3b8;}
.section-header{font-size:1.3rem;font-weight:700;color:#e2e8f0;margin:2rem 0 1rem 0;display:flex;align-items:center;gap:0.5rem;}
.section-header::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.5),transparent);margin-left:0.75rem;}
.nb-card{background:rgba(255,255,255,0.03);border:1px solid rgba(124,58,237,0.25);border-radius:16px;padding:1.5rem;margin-bottom:1rem;transition:all 0.3s;position:relative;overflow:hidden;}
.nb-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#7C3AED,#06B6D4);}
.nb-card:hover{border-color:rgba(124,58,237,0.6);background:rgba(124,58,237,0.06);transform:translateX(4px);}
.nb-num{font-size:2rem;font-weight:800;color:rgba(124,58,237,0.3);position:absolute;right:1.25rem;top:1rem;}
.nb-title{font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:0.4rem;padding-right:3rem;}
.nb-desc{font-size:0.875rem;color:#94a3b8;line-height:1.65;margin-bottom:0.75rem;}
.nb-tag{display:inline-block;background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);border-radius:50px;padding:2px 10px;font-size:0.72rem;color:#67e8f9;margin:2px;}
.tech-card{background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.2);border-radius:14px;padding:1.25rem;text-align:center;transition:all 0.3s;backdrop-filter:blur(10px);}
.tech-card:hover{border-color:rgba(124,58,237,0.5);transform:translateY(-4px);box-shadow:0 10px 25px rgba(124,58,237,0.2);}
.tech-icon{font-size:2.2rem;margin-bottom:0.5rem;}
.tech-name{font-size:0.9rem;font-weight:700;color:#e2e8f0;margin-bottom:0.2rem;}
.tech-role{font-size:0.75rem;color:#64748b;}
.pipeline-step{background:rgba(255,255,255,0.03);border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.5rem;display:flex;align-items:flex-start;gap:1rem;transition:all 0.3s;}
.pipeline-step:hover{border-color:rgba(124,58,237,0.4);background:rgba(124,58,237,0.05);}
.ps-num{background:linear-gradient(135deg,#7C3AED,#4F46E5);border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;color:white;flex-shrink:0;margin-top:0.1rem;}
.ps-title{font-size:0.9rem;font-weight:700;color:#e2e8f0;margin-bottom:0.2rem;}
.ps-desc{font-size:0.8rem;color:#64748b;line-height:1.5;}
.github-btn{display:inline-block;background:linear-gradient(135deg,#1f2937,#374151);border:1px solid rgba(255,255,255,0.15);border-radius:10px;padding:0.65rem 1.5rem;color:#e2e8f0;text-decoration:none;font-weight:600;font-size:0.9rem;transition:all 0.3s;margin:0.25rem;}
.github-btn:hover{background:linear-gradient(135deg,#374151,#4B5563);transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.3);}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0 1.5rem;'><div style='font-size:2.8rem;'>📖</div><div style='font-size:1rem;font-weight:700;color:#a78bfa;margin-top:0.3rem;'>About</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    st.page_link("streamlit_app.py",            label="🏠 Home / Dashboard")
    st.page_link("pages/1_Single_Analysis.py",  label="✍️ Single Review Analysis")
    st.page_link("pages/2_Batch_Analysis.py",   label="📂 Batch File Analysis")
    st.page_link("pages/3_Model_Comparison.py", label="📊 Model Comparison")
    st.page_link("pages/4_About.py",            label="📖 About & Notebooks")

results = load_results()

st.markdown("""
<div class="page-header">
    <div class="page-title">📖 About This Project</div>
    <div class="page-sub">
        A comprehensive NLP research project exploring multiple approaches to binary
        sentiment classification on the Amazon product review dataset — from classical
        bag-of-words models to deep learning and transformer-based transfer learning.
    </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🎯 Problem Statement</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(124,58,237,0.2);border-radius:14px;padding:1.5rem 2rem;font-size:0.9rem;color:#94a3b8;line-height:1.8;">
    <p>Online product reviews carry enormous commercial value — they influence purchasing decisions,
    highlight product flaws, and shape brand perception. However, manually reading and categorising
    thousands of reviews is infeasible at scale.</p>
    <p>This project investigates <b style="color:#a78bfa;">automated sentiment analysis</b> —
    the task of automatically classifying review text as <b style="color:#10B981;">Positive</b>
    or <b style="color:#EF4444;">Negative</b> — using a progression of NLP techniques ranging
    from classical machine learning to modern deep learning.</p>
    <p style="margin-bottom:0;"><b style="color:#e2e8f0;">Dataset:</b> 1,000 labelled Amazon product reviews across electronics &amp; accessories
    categories, balanced with approximately 50% positive and 50% negative samples.</p>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">📓 Research Notebooks</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;font-size:0.875rem;margin-bottom:1.25rem;'>"
    "Four Jupyter notebooks explore progressively more complex NLP approaches, "
    "each building on the insights of the previous.</p>",
    unsafe_allow_html=True,
)

for i, nb in enumerate(NOTEBOOK_META, 1):
    tags_html = "".join(f'<span class="nb-tag">{t}</span>' for t in nb["tags"])
    st.markdown(f"""
    <div class="nb-card">
        <div class="nb-num">{i:02d}</div>
        <div class="nb-title">📓 {nb['title']}</div>
        <div class="nb-desc">{nb['desc']}</div>
        {tags_html}
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">⚙️ ML Pipeline</div>', unsafe_allow_html=True)

pipeline_steps = [
    ("Data Ingestion",      "Raw CSV dataset loaded from `data/raw/amazonLabelled.csv`. Columns: Serial, Feedback (text), Sentiment (Positive/Negative)."),
    ("Preprocessing",       "Text lowercased, whitespace collapsed, and empty rows removed via `src/preprocessing.py`. Sentiment mapped to binary labels (1/0)."),
    ("Feature Extraction",  "CountVectorizer with English stop-word removal transforms raw text into sparse TF feature matrices (vocabulary ~3,000 tokens)."),
    ("Model Training",      "Three classifiers trained on 80% of data: Logistic Regression, Random Forest (100 trees), Multinomial Naive Bayes."),
    ("Evaluation",          "Accuracy evaluated on the held-out 20% test split. Models + vectorizer serialised to `outputs/models/*.pkl`."),
    ("Serving",             "Streamlit app loads pickled models, preprocesses user input identically to training, and returns real-time predictions with confidence."),
]

for i, (title, desc) in enumerate(pipeline_steps, 1):
    st.markdown(f"""
    <div class="pipeline-step">
        <div class="ps-num">{i}</div>
        <div>
            <div class="ps-title">{title}</div>
            <div class="ps-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)

tech_items = [
    ("🐍", "Python 3.11",     "Core language"),
    ("🤖", "scikit-learn",    "ML models & vectorizer"),
    ("🧮", "NumPy / Pandas",  "Data manipulation"),
    ("🌊", "Streamlit",       "Web application"),
    ("📈", "Plotly",          "Interactive charts"),
    ("🧠", "TensorFlow/Keras","Deep learning (LSTM)"),
    ("📝", "Gensim",          "Word2Vec embeddings"),
    ("🤗", "Transformers",    "Transfer learning"),
    ("📓", "Jupyter",         "Research notebooks"),
]
tc_cols = st.columns(3)
for i, (icon, name, role) in enumerate(tech_items):
    with tc_cols[i % 3]:
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-icon">{icon}</div>
            <div class="tech-name">{name}</div>
            <div class="tech-role">{role}</div>
        </div>""", unsafe_allow_html=True)
    if (i + 1) % 3 == 0:
        st.markdown("")

st.markdown('<div class="section-header">🎯 Deployed Model Results</div>', unsafe_allow_html=True)
import pandas as pd
summary_rows = []
for model_name, key in ACCURACY_DISPLAY.items():
    meta = MODEL_META.get(model_name, {})
    acc  = results.get(key, 0)
    summary_rows.append({
        "Model":        f"{meta.get('icon','')} {model_name}",
        "Type":         meta.get("type",""),
        "Vectorizer":   meta.get("vectorizer",""),
        "Speed":        meta.get("speed",""),
        "Test Accuracy":f"{round(acc*100,1)}%",
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown('<div class="section-header">💡 Key Findings</div>', unsafe_allow_html=True)
findings = [
    ("📊", "Naive Bayes leads", "Multinomial NB achieved the highest test accuracy (78.5%) despite being the simplest probabilistic model — demonstrating that word count distributions are highly informative for this binary task."),
    ("⚖️", "Classical ML is competitive", "Logistic Regression and Random Forest performed comparably (76.5% and 76.0%), suggesting the dataset may benefit more from better features than more complex classifiers."),
    ("🧠", "Deep learning advantages", "LSTM and BiLSTM models (in notebooks) capture sequential context that bag-of-words models miss — particularly for sarcasm and nuanced sentiment."),
    ("🚀", "Transfer learning ceiling", "Fine-tuned BERT-style models (notebook 4) approach SOTA performance but require significantly more compute and data to show clear advantages on this small 1,000-sample dataset."),
    ("🔤", "Feature engineering matters", "Word2Vec embeddings (notebook 2) provide semantic similarity that CountVectorizer lacks — e.g., 'terrible' and 'awful' are understood as synonymous rather than independent features."),
]
for icon, title, body in findings:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(124,58,237,0.2);border-left:4px solid #7C3AED;border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
        <div style="font-size:0.9rem;font-weight:700;color:#e2e8f0;margin-bottom:0.2rem;">{icon} {title}</div>
        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">{body}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1.5rem 0;color:#334155;font-size:0.8rem;">
    Built with ❤️ using <b style="color:#a78bfa;">Streamlit</b> · 
    scikit-learn · Plotly · TensorFlow · Gensim · Hugging Face Transformers<br>
    <span style="color:#475569;">Amazon Review Sentiment Analysis — NLP Research Project</span>
</div>""", unsafe_allow_html=True)

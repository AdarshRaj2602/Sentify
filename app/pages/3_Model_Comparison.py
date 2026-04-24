# Side-by-side model performance and comparison

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_models, load_vectorizer, load_results, load_raw_dataset,
    load_transfer_model, load_css,
    predict_sentiment, MODEL_META, ACCURACY_DISPLAY,
    TL_SST2_ACCURACY, TL_MODEL_NAME,
    sentiment_color, sentiment_emoji,
)

st.set_page_config(
    page_title="Model Comparison · Amazon Sentiment",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded",
)

load_css("model_comparison.css")

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0 1.5rem;'><div style='font-size:2.8rem;'>📊</div><div style='font-size:1rem;font-weight:700;color:#4ade80;margin-top:0.3rem;'>Model Comparison</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    st.page_link("streamlit_app.py",            label="🏠 Home / Dashboard")
    st.page_link("pages/1_Single_Analysis.py",  label="✍️ Single Review Analysis")
    st.page_link("pages/2_Batch_Analysis.py",   label="📂 Batch File Analysis")
    st.page_link("pages/3_Model_Comparison.py", label="📊 Model Comparison")
    st.page_link("pages/4_About.py",            label="📖 About & Notebooks")

models     = load_models()
vectorizer = load_vectorizer()
results    = load_results()
tl_model   = load_transfer_model()   # None if transformers not installed

st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Model Comparison</div>
    <div class="page-sub">
        Deep-dive into each model's accuracy, characteristics, and real-time head-to-head predictions.
        Understand which model is best suited for your use case.
    </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">🏆 Accuracy Leaderboard</div>', unsafe_allow_html=True)

# Build leaderboard: classical models from results.json + TL with SST-2 benchmark
leaderboard_entries = [
    (name, key, results.get(key, 0), False)
    for name, key in ACCURACY_DISPLAY.items()
    if key != "__transfer__"
]
# Add TL entry
leaderboard_entries.append((TL_MODEL_NAME, "__transfer__", TL_SST2_ACCURACY, True))
# Sort by accuracy descending
leaderboard_entries.sort(key=lambda x: x[2], reverse=True)

card_cols = st.columns(len(leaderboard_entries))
for rank, (col_w, (model_name, key, acc, is_tl)) in enumerate(zip(card_cols, leaderboard_entries)):
    meta       = MODEL_META.get(model_name, {})
    rank_emoji = ["\U0001f947", "\U0001f948", "\U0001f949", "4️⃣"][min(rank, 3)]
    is_winner  = (rank == 0)
    acc_label  = f"{round(acc*100,1)}%"
    note_html  = "<div style='font-size:0.68rem;color:#F59E0B;margin-top:0.4rem;'>* SST-2 benchmark</div>" if is_tl else ""
    border_style = (
        "border-color:rgba(245,158,11,0.6);box-shadow:0 8px 30px rgba(245,158,11,0.2);"
        if is_winner else
        ("border-color:rgba(251,191,36,0.4);" if is_tl else "")
    )
    with col_w:
        winner_html  = '<span class="winner-badge">👑 Best</span>' if is_winner else ""
        tl_badge     = ('<span style="background:linear-gradient(135deg,#F59E0B,#EF4444);border-radius:50px;'
                        'padding:2px 8px;font-size:0.65rem;font-weight:700;color:white;margin-left:4px;">'
                        '🤗 TL</span>') if is_tl else ""
        note_html    = ('<div style="font-size:0.68rem;color:#F59E0B;margin-top:0.4rem;">'
                        '* SST-2 benchmark</div>') if is_tl else ""
        acc_label_txt = "SST-2 Benchmark" if is_tl else "Test Set Accuracy"
        model_type   = meta.get("type", "Classifier")
        model_speed  = meta.get("speed", "N/A")
        model_icon   = meta.get("icon", "🔷")
        card_html = (
            f'<div class="compare-card" style="{border_style}">'
            f'<div style="font-size:1.8rem;margin-bottom:0.5rem;">{rank_emoji} {model_icon}</div>'
            f'<div class="card-model-name">{model_name}{winner_html}{tl_badge}</div>'
            f'<div class="card-model-type">{model_type}</div>'
            f'<div class="card-accuracy">{acc_label}</div>'
            f'<div class="card-acc-label">{acc_label_txt}</div>'
            f'{note_html}'
            f'<div style="margin-top:0.75rem;font-size:0.8rem;color:#475569;">'
            f'Speed: <span style="color:#a78bfa;">{model_speed}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown('<div class="section-header">📈 Accuracy Comparison</div>', unsafe_allow_html=True)

bar_names  = [e[0] for e in leaderboard_entries]
bar_accs   = [round(e[2]*100,1) for e in leaderboard_entries]
bar_is_tl  = [e[3] for e in leaderboard_entries]

# Use a distinct colour palette
palette = ["#7C3AED", "#4F46E5", "#06B6D4", "#F59E0B"]
bars_palette = []
ci = 0
for is_tl_flag in bar_is_tl:
    bars_palette.append("#F59E0B" if is_tl_flag else palette[ci % 3])
    if not is_tl_flag:
        ci += 1

fig_acc = go.Figure()
for i, (name, acc, color) in enumerate(zip(bar_names, bar_accs, bars_palette)):
    fig_acc.add_trace(go.Bar(
        x=[name], y=[acc],
        marker=dict(color=color, line=dict(width=0)),
        text=[f"{acc}%"], textposition="outside",
        textfont=dict(color="white", size=14, family="Inter"),
        width=0.4,
        hovertemplate=f"<b>{name}</b><br>Accuracy: {acc}%<extra></extra>",
    ))
# Reference line at 80%
fig_acc.add_hline(y=80, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  annotation_text="80% baseline", annotation_font_color="#475569",
                  annotation_font_family="Inter")
fig_acc.add_annotation(
    x=TL_MODEL_NAME, y=TL_SST2_ACCURACY * 100 + 4,
    text="SST-2 benchmark",
    showarrow=False, font=dict(color="#F59E0B", size=10, family="Inter"),
    xanchor="center",
)
fig_acc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"), showlegend=False,
    xaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(family="Inter", size=13)),
    yaxis=dict(range=[0,110], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
               ticksuffix="%", color="#94a3b8", tickfont=dict(family="Inter")),
    margin=dict(t=20,b=20,l=50,r=20), height=320, bargap=0.5,
)
st.plotly_chart(fig_acc, use_container_width=True)

st.markdown('<div class="section-header">🕷️ Model Attribute Radar</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;'>"
    "Qualitative comparison across five key dimensions (1 = lowest, 5 = highest). "
    "Scores are based on well-established characteristics of each algorithm class.</p>",
    unsafe_allow_html=True,
)

categories  = ["Accuracy", "Speed", "Interpretability", "Robustness", "Scalability"]
radar_data  = {
    "Logistic Regression":          [3.8, 5.0, 5.0, 3.5, 4.5],
    "Random Forest":                 [3.8, 2.5, 2.5, 4.5, 3.0],
    "Naive Bayes":                   [3.9, 5.0, 4.0, 3.0, 5.0],
    "Transfer Learning (DistilBERT)":[4.6, 1.5, 2.0, 5.0, 2.0],
}
radar_colors = ["#7C3AED", "#06B6D4", "#4F46E5", "#F59E0B"]

fig_radar = go.Figure()
for (model_name, vals), color in zip(radar_data.items(), radar_colors):
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        name=model_name, line=dict(color=color, width=2),
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.1,)}",
        hovertemplate="<b>%{theta}</b>: %{r}/5<extra>" + model_name + "</extra>",
    ))
fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0,5], tickfont=dict(color="#475569", family="Inter", size=9),
                        gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
        angularaxis=dict(tickfont=dict(color="#94a3b8", family="Inter", size=11),
                         gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    legend=dict(font=dict(color="#e2e8f0", family="Inter"), bgcolor="rgba(0,0,0,0)",
                orientation="h", y=-0.15),
    margin=dict(t=20,b=60,l=60,r=60), height=380,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown('<div class="section-header">⚔️ Head-to-Head Live Test</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;'>"
    "Enter a review and see all models fight it out in real time.</p>",
    unsafe_allow_html=True,
)

test_input = st.text_area(
    "h2h_input", height=100,
    placeholder="Type a review to compare all models head-to-head...",
    label_visibility="collapsed",
)
h2h_col, _ = st.columns([1,3])
with h2h_col:
    h2h_run = st.button("⚔️  Compare Models", use_container_width=True)

if h2h_run:
    if not test_input.strip():
        st.warning("⚠️  Please enter some review text.")
    else:
        all_h2h = dict(models)  # classical
        if tl_model:
            all_h2h[TL_MODEL_NAME] = tl_model

        h2h_results = {
            name: predict_sentiment(test_input, mdl, vectorizer)
            for name, mdl in all_h2h.items()
        }

        # Comparison table
        rows = []
        for name, res in h2h_results.items():
            acc_key = ACCURACY_DISPLAY.get(name,"")
            rows.append({
                "Model":      name,
                "Prediction": res["label"],
                "Confidence": f"{round(res['confidence']*100,1)}%" if res["confidence"] else "N/A",
                "Test Accuracy": f"{round(results.get(acc_key,0)*100,1) if acc_key != '__transfer__' else TL_SST2_ACCURACY*100}%",
                "Speed":      MODEL_META.get(name,{}).get("speed",""),
            })
        cmp_df = pd.DataFrame(rows)

        def color_pred(val):
            if val == "Positive": return "color:#10B981;font-weight:700;"
            if val == "Negative": return "color:#EF4444;font-weight:700;"
            return ""

        styled_cmp = cmp_df.style.map(color_pred, subset=["Prediction"])
        st.dataframe(styled_cmp, use_container_width=True, hide_index=True)

        # Agreement visual
        labels = [r["Prediction"] for r in h2h_results.values()]
        pos_v  = labels.count("Positive")
        neg_v  = labels.count("Negative")
        if pos_v == len(labels):
            st.success(f"✅ **Unanimous: Positive** — all {len(labels)} models agree.")
        elif neg_v == len(labels):
            st.error(f"🔴 **Unanimous: Negative** — all {len(labels)} models agree.")
        else:
            st.warning(f"⚠️ **Split verdict** — {pos_v} Positive / {neg_v} Negative")

st.markdown('<div class="section-header">🗂️ Technical Specifications</div>', unsafe_allow_html=True)

spec_data = {
    "Property":              ["Algorithm family",   "Vectorizer",        "Supports proba", "Handles imbalance",   "Interpretable",  "Training speed", "Inference speed"],
    "Logistic Regression":   ["Linear",             "CountVectorizer",   "✅ Yes",          "With class_weight",   "✅ High",          "⚡ Fast",         "⚡ Fastest"     ],
    "Random Forest":         ["Ensemble tree",      "CountVectorizer",   "✅ Yes",          "With class_weight",   "❌ Low",           "🐢 Slow",         "🐢 Moderate"    ],
    "Naive Bayes":           ["Probabilistic",      "CountVectorizer",   "✅ Yes",          "With priors",         "🟡 Medium",       "⚡ Fastest",      "⚡ Fastest"     ],
    "Transfer Learning":     ["Transformer (BERT)", "WordPiece tokenizer","✅ Yes",          "Via sampling",        "🟡 Attention maps","⚠️ Pre-trained",  "🐌 Slowest"     ],
}
spec_df = pd.DataFrame(spec_data).set_index("Property")
st.dataframe(spec_df, use_container_width=True)

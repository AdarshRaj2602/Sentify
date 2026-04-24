# Single review sentiment analysis page

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import streamlit as st
import plotly.graph_objects as go
from utils import (
    load_models, load_vectorizer, load_results, load_transfer_model,
    predict_sentiment, clean_text, load_css,
    MODEL_META, ACCURACY_DISPLAY, TL_SST2_ACCURACY, TL_MODEL_NAME,
    sentiment_emoji, sentiment_color,
)

st.set_page_config(
    page_title="Single Review Analysis · Amazon Sentiment",
    page_icon="✍️", layout="wide", initial_sidebar_state="expanded",
)

load_css("single_analysis.css")

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0 1.5rem;'><div style='font-size:2.8rem;'>✍️</div><div style='font-size:1rem;font-weight:700;color:#a78bfa;margin-top:0.3rem;'>Single Analysis</div></div>", unsafe_allow_html=True)
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
tl_model   = load_transfer_model()

st.markdown("""
<div class="page-header">
    <div class="page-title">✍️ Single Review Analysis</div>
    <div class="page-sub">
        Type or paste any Amazon product review and instantly compare predictions across
        all trained models — with confidence scores and a majority-vote verdict.
    </div>
</div>""", unsafe_allow_html=True)

EXAMPLES = {
    "😊 Glowing positive": "This is absolutely the best product I have ever bought! The quality is superb and it arrived quickly. Highly recommend to everyone.",
    "😠 Clear negative":   "Total waste of money. Broke after two days, terrible build quality, and customer service was completely useless. Avoid at all costs.",
    "🤔 Mixed / nuanced":  "The sound quality is decent for the price, but the battery drains faster than expected and the fit isn't comfortable for long sessions.",
    "📦 Short review":     "Works great!",
    "💬 Sarcastic":        "Oh sure, because what I really wanted was a product that stops working the moment the warranty expires.",
}
st.markdown('<div class="section-header">💡 Try an Example</div>', unsafe_allow_html=True)
selected_example = st.selectbox(
    "Pick an example or type your own below:",
    options=["— type your own —"] + list(EXAMPLES.keys()),
    label_visibility="collapsed",
)
default_text = EXAMPLES.get(selected_example, "") if selected_example != "— type your own —" else ""

st.markdown('<div class="section-header">📝 Review Text</div>', unsafe_allow_html=True)
user_text = st.text_area(
    "review_input", value=default_text, height=140,
    placeholder="Paste or type an Amazon product review here...",
    label_visibility="collapsed",
)
wc = len(user_text.split()) if user_text.strip() else 0
st.markdown(f"<div style='font-size:0.78rem;color:#475569;text-align:right;margin-top:-0.4rem;'>{wc} words · {len(user_text.strip())} chars</div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">🤖 Select Models</div>', unsafe_allow_html=True)
model_check_cols = st.columns(4)
selected_models  = {}
for i, model_name in enumerate(models.keys()):
    icon = MODEL_META.get(model_name, {}).get("icon", "🔷")
    with model_check_cols[i]:
        if st.checkbox(f"{icon} {model_name}", value=True, key=f"chk_{model_name}"):
            selected_models[model_name] = models[model_name]

with model_check_cols[3]:
    tl_enabled = tl_model is not None
    tl_help = "Downloads ~250 MB on first use. Runs on CPU — may take a few seconds per review." if tl_enabled else "Install transformers to enable: pip install transformers"
    tl_checked = st.checkbox(
        f"🤗 {TL_MODEL_NAME}",
        value=False,
        key="chk_tl",
        disabled=not tl_enabled,
        help=tl_help,
    )
    if tl_checked and tl_enabled:
        selected_models[TL_MODEL_NAME] = tl_model
    if not tl_enabled:
        st.markdown("<span style='font-size:0.72rem;color:#EF4444;'>Install transformers to enable</span>", unsafe_allow_html=True)
    elif tl_checked:
        st.markdown("<span style='font-size:0.72rem;color:#F59E0B;'>⚠️ Slower — ~250 MB model</span>", unsafe_allow_html=True)

st.markdown("")
btn_col, _ = st.columns([1, 3])
with btn_col:
    run = st.button("🚀  Analyse Sentiment", use_container_width=True)

if run:
    if not user_text.strip():
        st.warning("⚠️  Please enter a review before running analysis.")
    elif not selected_models:
        st.warning("⚠️  Please select at least one model.")
    else:
        with st.spinner("Running inference..."):
            time.sleep(0.3)
            all_results = {
                name: predict_sentiment(user_text, mdl, vectorizer)
                for name, mdl in selected_models.items()
            }

        st.markdown('<div class="section-header">📊 Model Predictions</div>', unsafe_allow_html=True)
        result_cols = st.columns(len(all_results))
        labels_list = []

        for col, (model_name, res) in zip(result_cols, all_results.items()):
            label    = res["label"]
            conf     = res["confidence"]
            conf_pct = round(conf * 100, 1) if conf else None
            # Transfer learning gets amber/gold card; others get standard colouring
            is_tl   = (model_name == TL_MODEL_NAME)
            css_cls = "transfer" if is_tl else ("positive" if label == "Positive" else "negative")
            color   = "#F59E0B" if is_tl else sentiment_color(label)
            emoji   = sentiment_emoji(label)
            labels_list.append(label)

            tl_badge_html = (
                '<div style="font-size:0.68rem;color:#F59E0B;margin-bottom:0.4rem;">'
                '🚀 Transfer Learning &middot; DistilBERT</div>'
            ) if is_tl else ""
            conf_display = f"{conf_pct}%" if conf_pct is not None else "N/A"
            model_icon   = MODEL_META.get(model_name, {}).get("icon", "🔷")
            card_html = (
                f'<div class="result-card {css_cls}">'
                f'<div class="rc-model-name">{model_icon} {model_name}</div>'
                f'{tl_badge_html}'
                f'<span class="rc-emoji">{emoji}</span>'
                f'<div class="rc-label" style="color:{color};">{label}</div>'
                f'<div class="rc-conf">Confidence: <b style="color:{color};">{conf_display}</b></div>'
                f'</div>'
            )

            with col:
                st.markdown(card_html, unsafe_allow_html=True)

                if conf_pct is not None:
                    gauge_color = "#F59E0B" if is_tl else color
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf_pct,
                        number=dict(suffix="%", font=dict(color=gauge_color, family="Inter", size=22)),
                        gauge=dict(
                            axis=dict(range=[0,100], tickfont=dict(color="#475569", family="Inter", size=9)),
                            bar=dict(color=gauge_color, thickness=0.3),
                            bgcolor="rgba(0,0,0,0)", borderwidth=0,
                            steps=[
                                dict(range=[0,50],  color="rgba(255,255,255,0.03)"),
                                dict(range=[50,75], color="rgba(255,255,255,0.05)"),
                                dict(range=[75,100],color="rgba(255,255,255,0.07)"),
                            ],
                            threshold=dict(line=dict(color=gauge_color, width=3), thickness=0.8, value=conf_pct),
                        ),
                    ))
                    fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#94a3b8"),
                        height=160, margin=dict(t=10,b=5,l=10,r=10),
                    )
                    st.plotly_chart(fig_g, use_container_width=True, key=f"gauge_{model_name}")

        pos_v = labels_list.count("Positive")
        neg_v = labels_list.count("Negative")
        total = len(labels_list)
        if pos_v > neg_v:
            verdict, v_emoji, v_class, v_color = "Positive", "🟢", "positive", "#10B981"
            agree = f"{pos_v}/{total} models agree"
        elif neg_v > pos_v:
            verdict, v_emoji, v_class, v_color = "Negative", "🔴", "negative", "#EF4444"
            agree = f"{neg_v}/{total} models agree"
        else:
            verdict, v_emoji, v_class, v_color = "Neutral (Tie)", "🟡", "tie", "#F59E0B"
            agree = "Models are split equally"

        st.markdown(f"""
        <div class="verdict-box {v_class}">
            <div class="verdict-label">🏆 Majority Verdict</div>
            <div class="verdict-value" style="color:{v_color};">{v_emoji} {verdict}</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.4rem;">{agree}</div>
        </div>""", unsafe_allow_html=True)

        conf_data = {n: (r["confidence"] or 0)*100 for n,r in all_results.items()}
        if any(v > 0 for v in conf_data.values()):
            st.markdown('<div class="section-header" style="margin-top:2rem;">📈 Confidence Comparison</div>', unsafe_allow_html=True)
            bar_palette = ["#7C3AED", "#4F46E5", "#06B6D4", "#F59E0B"]
            fig_c = go.Figure()
            for i,(name,cval) in enumerate(conf_data.items()):
                bar_color = "#F59E0B" if name == TL_MODEL_NAME else bar_palette[i % 3]
                fig_c.add_trace(go.Bar(
                    x=[name], y=[round(cval,1)],
                    marker=dict(color=bar_color, line=dict(width=0)),
                    text=[f"{round(cval,1)}%"], textposition="outside",
                    textfont=dict(color="white", size=13, family="Inter"),
                    width=0.45,
                    hovertemplate=f"<b>{name}</b><br>Confidence: {round(cval,1)}%<extra></extra>",
                ))
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"), showlegend=False,
                xaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(family="Inter")),
                yaxis=dict(range=[0,115], showgrid=True,
                           gridcolor="rgba(255,255,255,0.05)",
                           ticksuffix="%", color="#94a3b8", tickfont=dict(family="Inter")),
                margin=dict(t=20,b=20,l=40,r=20), height=250, bargap=0.4,
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("🔍 View preprocessed input text"):
            cleaned = clean_text(user_text)
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(124,58,237,0.2);"
                f"border-radius:10px;padding:1rem;font-size:0.88rem;color:#94a3b8;font-family:monospace;'>"
                f"{cleaned}</div>",
                unsafe_allow_html=True,
            )

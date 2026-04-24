# Batch file upload - CSV, Excel, TXT support

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_models, load_vectorizer, batch_predict, load_transfer_model, load_css,
    MODEL_META, sentiment_color, TL_MODEL_NAME,
)

st.set_page_config(
    page_title="Batch Analysis · Amazon Sentiment",
    page_icon="📂", layout="wide", initial_sidebar_state="expanded",
)

load_css("batch_analysis.css")

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0 1.5rem;'><div style='font-size:2.8rem;'>📂</div><div style='font-size:1rem;font-weight:700;color:#67e8f9;margin-top:0.3rem;'>Batch Analysis</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    st.page_link("streamlit_app.py",            label="🏠 Home / Dashboard")
    st.page_link("pages/1_Single_Analysis.py",  label="✍️ Single Review Analysis")
    st.page_link("pages/2_Batch_Analysis.py",   label="📂 Batch File Analysis")
    st.page_link("pages/3_Model_Comparison.py", label="📊 Model Comparison")
    st.page_link("pages/4_About.py",            label="📖 About & Notebooks")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#475569;line-height:1.8;'>
        <b style='color:#67e8f9;'>Supported Formats</b><br>
        📄 CSV (.csv)<br>
        📊 Excel (.xlsx, .xls)<br>
        📝 Text (.txt, one per line)<br><br>
        <b style='color:#67e8f9;'>Max File Size</b><br>
        200 MB (Streamlit default)
    </div>""", unsafe_allow_html=True)

models     = load_models()
vectorizer = load_vectorizer()
tl_model   = load_transfer_model()   # None if transformers not installed

st.markdown("""
<div class="page-header">
    <div class="page-title">📂 Batch File Analysis</div>
    <div class="page-sub">
        Upload a CSV, Excel, or plain-text file containing product reviews. All selected models
        will score every row simultaneously — then download your results.
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <span class='format-pill'>📄 CSV</span>
    <span class='format-pill'>📊 XLSX / XLS</span>
    <span class='format-pill'>📝 TXT</span>
    <span class='format-pill'>⚡ Multi-model inference</span>
    <span class='format-pill'>⬇️ Downloadable results</span>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-header">📤 Upload Your File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload file",
    type=["csv", "xlsx", "xls", "txt"],
    label_visibility="collapsed",
    help="CSV / Excel: must contain a column with review text. TXT: one review per line.",
)

df_raw = None
file_type = None

if uploaded_file is not None:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    file_type = ext

    try:
        if ext == "csv":
            df_raw = pd.read_csv(uploaded_file)
        elif ext in ("xlsx", "xls"):
            df_raw = pd.read_excel(uploaded_file, engine="openpyxl" if ext == "xlsx" else None)
        elif ext == "txt":
            content = uploaded_file.read().decode("utf-8", errors="replace")
            lines   = [l.strip() for l in content.splitlines() if l.strip()]
            df_raw  = pd.DataFrame({"review": lines})
        st.success(f"✅ **{uploaded_file.name}** loaded — {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        df_raw = None

if df_raw is not None:
    st.markdown('<div class="section-header">👁️ File Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df_raw.head(5),
        use_container_width=True,
        hide_index=True,
    )

    # Choose text column
    text_columns = df_raw.select_dtypes(include="object").columns.tolist()
    if not text_columns:
        st.error("❌ No text columns found. Please upload a file with string/text data.")
    else:
        st.markdown('<div class="section-header">🗂️ Select Text Column</div>', unsafe_allow_html=True)
        # Auto-detect common review column names
        default_col_candidates = [c for c in text_columns if any(
            kw in c.lower() for kw in ["review", "text", "feedback", "comment", "description"]
        )]
        default_idx = text_columns.index(default_col_candidates[0]) if default_col_candidates else 0

        text_col = st.selectbox(
            "Which column contains the review text?",
            options=text_columns,
            index=default_idx,
            label_visibility="collapsed",
        )

        st.info(f"ℹ️  Using column **\"{text_col}\"** as the review text source. "
                f"Non-empty rows: **{df_raw[text_col].dropna().shape[0]:,}**")

        # Model selection
        st.markdown('<div class="section-header">🤖 Select Models for Batch Inference</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        selected_batch_models = {}
        for col_widget, model_name in zip([mc1, mc2, mc3], models.keys()):
            icon = MODEL_META.get(model_name,{}).get("icon","🔷")
            with col_widget:
                if st.checkbox(f"{icon} {model_name}", value=True, key=f"batch_chk_{model_name}"):
                    selected_batch_models[model_name] = models[model_name]

        # TL option in 4th column
        tl_enabled = tl_model is not None
        with mc4:
            tl_batch_checked = st.checkbox(
                f"🤗 {TL_MODEL_NAME}",
                value=False,
                key="batch_chk_tl",
                disabled=not tl_enabled,
                help="Adds DistilBERT predictions. Significantly slower on large files (CPU inference).",
            )
            if tl_batch_checked and tl_enabled:
                selected_batch_models[TL_MODEL_NAME] = tl_model
            if not tl_enabled:
                st.markdown("<span style='font-size:0.72rem;color:#EF4444;'>pip install transformers</span>", unsafe_allow_html=True)
            elif tl_batch_checked:
                st.markdown("<span style='font-size:0.72rem;color:#F59E0B;'>⚠️ CPU — slow on large files</span>", unsafe_allow_html=True)

        st.markdown("")
        run_btn_col, _ = st.columns([1,3])
        with run_btn_col:
            run_batch = st.button("🚀  Run Batch Analysis", use_container_width=True)

        if run_batch:
            if not selected_batch_models:
                st.warning("⚠️  Please select at least one model.")
            else:
                total_rows = len(df_raw)
                prog_bar   = st.progress(0, text="Running inference...")
                try:
                    result_df = batch_predict(df_raw, text_col, selected_batch_models, vectorizer)
                    prog_bar.progress(100, text="Done!")

                    st.markdown('<div class="section-header">📊 Batch Results Summary</div>', unsafe_allow_html=True)

                    maj_col = "Majority_Sentiment"
                    pos_cnt = int((result_df[maj_col] == "Positive").sum())
                    neg_cnt = int((result_df[maj_col] == "Negative").sum())
                    tie_cnt = int((result_df[maj_col] == "Neutral (Tie)").sum())
                    pos_pct = round(pos_cnt / total_rows * 100, 1)
                    neg_pct = round(neg_cnt / total_rows * 100, 1)

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    for col_w, val, lbl in [
                        (sc1, f"{total_rows:,}", "Total Reviews"),
                        (sc2, f"{pos_cnt:,}",    "🟢 Positive"),
                        (sc3, f"{neg_cnt:,}",    "🔴 Negative"),
                        (sc4, f"{pos_pct}%",     "Positive Rate"),
                    ]:
                        with col_w:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{val}</div>
                                <div class="stat-label">{lbl}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("")
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        fig_pie = go.Figure(go.Pie(
                            labels=["Positive","Negative"] + (["Tie"] if tie_cnt else []),
                            values=[pos_cnt, neg_cnt] + ([tie_cnt] if tie_cnt else []),
                            hole=0.55,
                            marker=dict(colors=["#10B981","#EF4444","#F59E0B"],
                                        line=dict(color="rgba(0,0,0,0)",width=0)),
                            textfont=dict(family="Inter",size=13,color="white"),
                            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                        ))
                        fig_pie.add_annotation(
                            text=f"<b>{total_rows}</b><br>Reviews",
                            x=0.5,y=0.5,showarrow=False,
                            font=dict(size=13,color="#94a3b8",family="Inter"),
                            xanchor="center",yanchor="middle",
                        )
                        fig_pie.update_layout(
                            title=dict(text="Majority Sentiment Split",font=dict(color="#e2e8f0",size=14,family="Inter")),
                            paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter",color="#94a3b8"),
                            legend=dict(font=dict(color="#e2e8f0",family="Inter"),bgcolor="rgba(0,0,0,0)"),
                            margin=dict(t=50,b=20,l=20,r=20),height=300,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with ch2:
                        # Per-model positive rate
                        model_rates = {}
                        for mn in selected_batch_models:
                            col_name = f"{mn}_Sentiment"
                            if col_name in result_df.columns:
                                pos_r = round((result_df[col_name]=="Positive").mean()*100,1)
                                model_rates[mn] = pos_r

                        if model_rates:
                            fig_bar = go.Figure()
                            bar_colors = ["#7C3AED","#4F46E5","#06B6D4"]
                            for i,(mn,rate) in enumerate(model_rates.items()):
                                fig_bar.add_trace(go.Bar(
                                    x=[mn],y=[rate],
                                    marker=dict(color=bar_colors[i%3],line=dict(width=0)),
                                    text=[f"{rate}%"],textposition="outside",
                                    textfont=dict(color="white",size=12,family="Inter"),
                                    width=0.45,
                                    hovertemplate=f"<b>{mn}</b><br>Positive rate: {rate}%<extra></extra>",
                                ))
                            fig_bar.update_layout(
                                title=dict(text="Positive Rate per Model",font=dict(color="#e2e8f0",size=14,family="Inter")),
                                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter",color="#94a3b8"),showlegend=False,
                                xaxis=dict(showgrid=False,color="#94a3b8",tickfont=dict(family="Inter")),
                                yaxis=dict(range=[0,115],showgrid=True,gridcolor="rgba(255,255,255,0.05)",
                                           ticksuffix="%",color="#94a3b8",tickfont=dict(family="Inter")),
                                margin=dict(t=50,b=20,l=40,r=20),height=300,bargap=0.4,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown('<div class="section-header">📋 Detailed Results Table</div>', unsafe_allow_html=True)

                    # Colour-code majority column
                    def style_majority(val):
                        if val == "Positive":
                            return "color:#10B981;font-weight:700;"
                        elif val == "Negative":
                            return "color:#EF4444;font-weight:700;"
                        return "color:#F59E0B;font-weight:700;"

                    display_cols = [text_col] + \
                        [f"{mn}_Sentiment" for mn in selected_batch_models if f"{mn}_Sentiment" in result_df.columns] + \
                        [f"{mn}_Confidence_%" for mn in selected_batch_models if f"{mn}_Confidence_%" in result_df.columns] + \
                        ["Majority_Sentiment"]

                    styled = result_df[display_cols].style.map(
                        style_majority, subset=["Majority_Sentiment"]
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    st.markdown('<div class="section-header">⬇️ Download Results</div>', unsafe_allow_html=True)
                    dl1, dl2 = st.columns(2)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    with dl1:
                        st.download_button(
                            label="⬇️  Download as CSV",
                            data=csv_bytes,
                            file_name="sentiment_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Sentiment Results")
                    excel_bytes = excel_buf.getvalue()
                    with dl2:
                        st.download_button(
                            label="⬇️  Download as Excel",
                            data=excel_bytes,
                            file_name="sentiment_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Error during batch inference: {e}")
                    prog_bar.empty()

if df_raw is None:
    st.markdown('<div class="section-header" style="margin-top:2rem;">📖 How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Upload a File", "Drag & drop or browse a CSV, Excel, or TXT file containing your reviews."),
        ("2️⃣", "Select Text Column", "Choose which column holds the review text (auto-detected for common names)."),
        ("3️⃣", "Pick Your Models", "Select which ML models should score the data."),
        ("4️⃣", "Run & Download", "All models run in parallel. Download results as CSV or Excel when done."),
    ]
    step_cols = st.columns(4)
    for col_w, (num, title, desc) in zip(step_cols, steps):
        with col_w:
            st.markdown(f"""
            <div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);
            border-radius:14px;padding:1.25rem;text-align:center;height:100%;'>
                <div style='font-size:1.8rem;margin-bottom:0.5rem;'>{num}</div>
                <div style='font-size:0.9rem;font-weight:700;color:#67e8f9;margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.8rem;color:#64748b;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

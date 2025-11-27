import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import StringIO

# =============================================================================
# 1️⃣ TẢI HÀM XỬ LÝ PIPELINE
# =============================================================================
from config import QWEN_MODEL_PATH
import pipeline_ABSA as absa_pipeline

# =============================================================================
# 2️⃣ CUSTOM CSS
# =============================================================================
st.set_page_config(page_title="ABSA Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
    .main-header h1 { font-size: 3rem; font-weight: bold; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.75rem 2rem; font-size: 1.1rem; font-weight: bold; border-radius: 10px; width: 100%; }
    .stTextArea textarea { background: #2d2d2d !important; color: white !important; border-radius: 10px; border: 2px solid #555 !important; }
    .stDataFrame table, .stDataFrame th, .stDataFrame td { background-color: #2d2d2d !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3️⃣ HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>Hotel Aspect Based Sentiment Analysis Engine</h1>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 4️⃣ LOAD MODELS
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_models():
    return absa_pipeline.load_all_models()

try:
    models = get_models()
except Exception as e:
    st.error(f"❌ Critical error while loading models: {e}")
    st.stop()

def ensure_qwen_model_ready(models, announce=False):
    ready = bool(models and models.get("qwen_model"))
    if announce:
        if ready:
            st.success("✅ Qwen model is ready.")
        else:
            st.error("❌ Qwen model chưa sẵn sàng. Kiểm tra đường dẫn.")
    return ready

# =============================================================================
# 5️⃣ HÀM HIỂN THỊ DASHBOARD
# =============================================================================
def render_dashboard(results_df: pd.DataFrame):
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if {"category", "category_score"}.issubset(results_df.columns):
            st.markdown("### 📊 Điểm Category trung bình")
            avg_scores = results_df.dropna(subset=["category"]).groupby("category")["category_score"].mean().sort_values(ascending=False)
            if not avg_scores.empty:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.barh(avg_scores.index.tolist(), avg_scores.values.tolist(), color="#764ba2")
                ax.set_xlabel("Điểm trung bình")
                ax.invert_yaxis()
                for i, v in enumerate(avg_scores.values):
                    ax.text(v + 0.01, i, f"{v:.2f}", color="white", va="center")
                st.pyplot(fig)
            else:
                st.info("Chưa có dữ liệu category_score.")
        else:
            st.info("Thiếu cột category/category_score.")

    with chart_col2:
        if "polarity" in results_df.columns and not results_df["polarity"].empty:
            st.markdown("### 🥧 Tỷ lệ Polarity")
            polarity_share = results_df["polarity"].fillna("Unknown").value_counts(normalize=True).mul(100)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(polarity_share.values.tolist(), labels=polarity_share.index.tolist(), autopct="%1.1f%%", startangle=140, textprops={"color": "white"})
            ax.axis("equal")
            st.pyplot(fig)

    st.markdown("### ☁️ Word Cloud — Term & Opinion")
    col1, col2 = st.columns(2)

    def show_wordcloud(col, title, column_name):
        with col:
            st.markdown(f"#### {title}")
            if column_name in results_df.columns:
                text = " ".join(results_df[column_name].dropna().astype(str))
                if text.strip():
                    freq = pd.Series(text.lower().split()).value_counts().to_dict()
                    wc = WordCloud(width=500, height=300, background_color="black", colormap="viridis").generate_from_frequencies(freq)
                    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#2d2d2d')
                    ax.set_facecolor('#2d2d2d')
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info(f"No {column_name} data.")
            else:
                st.info(f"No {column_name} column.")
    
    show_wordcloud(col1, "🏷️ Term", "term")
    show_wordcloud(col2, "💬 Opinion", "opinion")

    if {"category", "polarity"}.issubset(results_df.columns):
        st.markdown("### 📊 Statistics by Category & Polarity")
        pivot = results_df.assign(polarity=results_df["polarity"].fillna("Unknown"), category=results_df["category"].fillna("Unknown")).groupby(["category", "polarity"]).size().unstack(fill_value=0)
        pivot["Total"] = pivot.sum(axis=1)
        st.dataframe(pivot.sort_index(), use_container_width=True)

    st.markdown("### 📋 Detailed results table")
    st.dataframe(results_df, use_container_width=True)
    st.download_button("💾 Download CSV results", results_df.to_csv(index=False).encode("utf-8"), "absa_results.csv", "text/csv")

# =============================================================================
# 6️⃣ TAB: SINGLE & BATCH
# =============================================================================
tab1, tab2 = st.tabs(["📝 Input review", "📤 Batch file analysis"])

# --- TAB 1 ---
with tab1:
    st.subheader("✍️ Enter review sentence")
    default_sentence = "The food was great and the staff was friendly, but the room was small and dirty."
    text_input = st.text_area("Enter content:", default_sentence, height=150)
    spacer, col1, col2 = st.columns([4, 1, 1])
    with col1:
        analyze = st.button("🔍 Analyze", use_container_width=True, key="single_btn")
    with col2:
        clear = st.button("🧹 Clear", use_container_width=True, key="clear_btn")
    if clear:
        st.rerun()

    if analyze and text_input.strip():
        if not ensure_qwen_model_ready(models, announce=True):
            st.error("❌ Qwen model is not ready. Please check the configuration / pipeline.")
        else:
            try:
                with st.spinner("🔄 Đang phân tích..."):
                    df = absa_pipeline.run_full_pipeline(text_input, models)
                if df.empty:
                    st.warning("No results from pipeline.")
                else:
                    st.success("✅ Analysis complete!")
                    render_dashboard(df)
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

# --- TAB 2 ---
with tab2:
    st.subheader("📂 Upload .txt file (each line is a review)")
    uploaded_file = st.file_uploader("Choose file", type=["txt"])
    if st.button("🚀 Run batch analysis", key="batch_btn"):
        if uploaded_file is None:
            st.warning("Please upload a file.")
        else:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                reviews = [line.strip() for line in stringio.readlines() if line.strip()]
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            if not reviews:
                st.warning("File is empty or invalid.")
            else:
                all_results = []
                st.info(f"Processing {len(reviews)} lines...")
                progress = st.progress(0)
                qwen_failed = False
                for i, review in enumerate(reviews):
                    try:
                        if not ensure_qwen_model_ready(models, announce=(i == 0)):
                            st.error("❌ Qwen model is not ready for batch processing. Stopping the process.")
                            qwen_failed = True
                            break
                        df = absa_pipeline.run_full_pipeline(review, models)
                        if not df.empty:
                            df["review_line"] = i + 1
                            df["review_text"] = review
                            all_results.append(df)
                    except Exception as e:
                        st.error(f"Lỗi dòng {i+1}: {e}")
                    progress.progress((i + 1) / len(reviews))
                progress.empty()

                if qwen_failed:
                    st.info("Please check the model configuration and try again.")
                elif not all_results:
                    st.warning("No aspects found.")
                else:
                    final_df = pd.concat(all_results, ignore_index=True)
                    st.success(f"✅ Completed! {len(final_df)} aspects found.")
                    render_dashboard(final_df)

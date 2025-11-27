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
import pipeline_ABSA as absa_pipeline

# =============================================================================
# 2️⃣ PAGE CONFIG & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Hotel Aspect Based Sentiment Analysis Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS cho giao diện modern và chuyên nghiệp
st.markdown(
    """
<style>
/* Google Fonts - Sử dụng font đẹp và hiện đại */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* CSS Variables cho theme nhất quán - LIGHT THEME */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --warning-gradient: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    
    --bg-primary: #f0f4f8;
    --bg-secondary: #e8eef5;
    --bg-card: rgba(255, 255, 255, 0.95);
    --bg-card-hover: rgba(255, 255, 255, 1);
    
    --text-primary: #1a202c;
    --text-secondary: #4a5568;
    --text-muted: #718096;
    
    --border-color: rgba(102, 126, 234, 0.15);
    --border-glow: rgba(102, 126, 234, 0.4);
    
    --shadow-soft: 0 4px 20px rgba(102, 126, 234, 0.1);
    --shadow-glow: 0 10px 40px rgba(102, 126, 234, 0.12);
}

/* Reset và base styles */
html, body, [class*="stApp"] {
    background: var(--bg-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

/* Background với pattern và gradient - LIGHT */
[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(ellipse at top left, rgba(102, 126, 234, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(118, 75, 162, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at center, rgba(79, 172, 254, 0.06) 0%, transparent 70%),
        linear-gradient(180deg, #f0f4f8 0%, #e8eef5 50%, #f5f7fa 100%);
    background-attachment: fixed;
}

/* Subtle geometric pattern for light theme */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Ẩn header mặc định của Streamlit */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 24px 60px;
    position: relative;
    z-index: 1;
}

/* ===== NAVBAR ===== */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid var(--border-color);
    margin-bottom: 24px;
    animation: slideDown 0.6s ease-out;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 14px;
}

.logo-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--primary-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
}

.logo-icon::before {
    content: "🏨";
    font-size: 22px;
    position: relative;
    z-index: 2;
}

.logo-icon::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 50%);
}

.logo-text {
    font-weight: 700;
    font-size: 20px;
    background: linear-gradient(135deg, #1a202c 0%, #4a5568 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.nav-badge {
    background: var(--primary-gradient);
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== HERO SECTION ===== */
.hero-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 48px 40px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-glow);
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--primary-gradient);
}

.hero-card::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(102, 126, 234, 0.12);
    border: 1px solid rgba(102, 126, 234, 0.25);
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    color: #667eea;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
}

.hero-eyebrow::before {
    content: "✨";
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 16px;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #1a202c 0%, #667eea 50%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 17px;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 600px;
}

.hero-features {
    display: flex;
    gap: 24px;
    margin-top: 28px;
    flex-wrap: wrap;
}

.hero-feature {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-secondary);
    font-size: 14px;
    font-weight: 500;
}

.hero-feature-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.hero-feature-icon.purple { background: rgba(102, 126, 234, 0.2); }
.hero-feature-icon.pink { background: rgba(240, 147, 251, 0.2); }
.hero-feature-icon.blue { background: rgba(79, 172, 254, 0.2); }

/* ===== TABS STYLING ===== */
.stTabs {
    animation: fadeInUp 0.8s ease-out 0.4s both;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--bg-card);
    padding: 8px;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(20px);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 12px 24px !important;
    background: transparent !important;
    border: none !important;
}

.stTabs button[role="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    transition: all 0.3s ease !important;
}

.stTabs button[role="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(102, 126, 234, 0.08) !important;
}

.stTabs button[aria-selected="true"] {
    background: var(--primary-gradient) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

/* ===== INPUT CARD ===== */
.input-section {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid var(--border-color);
    margin-top: 24px;
    transition: all 0.3s ease;
}

.input-section:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

/* Textarea styling */
.stTextArea textarea {
    background: #ffffff !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 18px 20px !important;
    transition: all 0.3s ease !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    background: #ffffff !important;
}

.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button[kind="primary"], 
.stButton > button:first-of-type {
    background: var(--primary-gradient) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button:first-of-type:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
}

.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: var(--border-glow) !important;
}

/* ===== RESULT SECTION ===== */
.result-section {
    margin-top: 32px;
    animation: fadeInUp 0.6s ease-out;
}

.result-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid var(--border-color);
    margin-bottom: 20px;
}

.result-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.result-card-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.result-card-icon.chart { background: rgba(102, 126, 234, 0.2); }
.result-card-icon.pie { background: rgba(240, 147, 251, 0.2); }
.result-card-icon.table { background: rgba(79, 172, 254, 0.2); }

.result-card-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
}

/* ===== DATAFRAME ===== */
.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
}

.stDataFrame table {
    background: #ffffff !important;
    border: none !important;
}

.stDataFrame th {
    background: rgba(102, 126, 234, 0.15) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 14px 16px !important;
    border: none !important;
}

.stDataFrame td {
    color: var(--text-secondary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.stDataFrame tr:hover td {
    background: rgba(102, 126, 234, 0.05) !important;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader {
    background: #ffffff !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    border-color: var(--border-glow) !important;
    background: rgba(102, 126, 234, 0.05) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
}

/* ===== ALERTS & MESSAGES ===== */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

.stSuccess {
    background: rgba(17, 153, 142, 0.15) !important;
    border-left: 4px solid #38ef7d !important;
}

.stWarning {
    background: rgba(242, 153, 74, 0.15) !important;
    border-left: 4px solid #f2994a !important;
}

.stError {
    background: rgba(245, 87, 108, 0.15) !important;
    border-left: 4px solid #f5576c !important;
}

.stInfo {
    background: rgba(79, 172, 254, 0.15) !important;
    border-left: 4px solid #4facfe !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div {
    background: var(--primary-gradient) !important;
    border-radius: 10px !important;
}

/* ===== SPINNER ===== */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton > button {
    background: var(--success-gradient) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(17, 153, 142, 0.4) !important;
}

/* ===== MATPLOTLIB CHARTS ===== */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background: transparent !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.5);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.7);
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .hero-title {
        font-size: 28px;
    }
    
    .hero-card {
        padding: 32px 24px;
    }
    
    .hero-features {
        flex-direction: column;
        gap: 16px;
    }
}

/* Block container padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 3️⃣ WRAPPER LAYOUT (NAVBAR + HERO)
# =============================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Navbar
st.markdown(
    """
<div class="top-nav">
    <div class="logo-section">
        <div class="logo-icon"></div>
        <span class="logo-text">Hotel ABSA Engine</span>
    </div>
    <div class="nav-badge">AI Powered</div>
</div>
""",
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
<div class="hero-card">
    <div class="hero-eyebrow">Aspect-Based Sentiment Analysis</div>
    <div class="hero-title">Hotel Aspect Based<br>Sentiment Analysis Engine</div>
    <div class="hero-subtitle">
        Transform hotel reviews into actionable insights. Our advanced ABSA engine analyzes 
        customer feedback across multiple aspects to help you understand what matters most.
    </div>
    <div class="hero-features">
        <div class="hero-feature">
            <div class="hero-feature-icon purple">🎯</div>
            <span>Aspect Detection</span>
        </div>
        <div class="hero-feature">
            <div class="hero-feature-icon pink">💭</div>
            <span>Sentiment Analysis</span>
        </div>
        <div class="hero-feature">
            <div class="hero-feature-icon blue">📊</div>
            <span>Visual Reports</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 4️⃣ LOAD MODELS
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_models():
    return absa_pipeline.load_all_models()

try:
    with st.spinner("🔄 Loading AI models..."):
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
    st.markdown('<div class="result-section">', unsafe_allow_html=True)

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if {"category", "category_score"}.issubset(results_df.columns):
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon chart">📊</div>
                        <div class="result-card-title">Category Score Overview</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            avg_scores = (
                results_df.dropna(subset=["category"])
                .groupby("category")["category_score"]
                .mean()
                .sort_values(ascending=False)
            )
            if not avg_scores.empty:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                # Create gradient-like bars
                colors = plt.cm.RdYlGn([(v + 1) / 2 for v in avg_scores.values])
                bars = ax.barh(avg_scores.index.tolist(), avg_scores.values.tolist(), color=colors, height=0.6)
                
                ax.set_xlabel("Average Score", color='#4a5568', fontsize=10)
                ax.tick_params(colors='#4a5568', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#e2e8f0')
                ax.spines['left'].set_color('#e2e8f0')
                ax.invert_yaxis()
                ax.set_xlim(-1.2, 1.2)
                
                for i, v in enumerate(avg_scores.values):
                    ax.text(v + 0.05 if v >= 0 else v - 0.15, i, f"{v:.2f}", 
                           va="center", color='#1a202c', fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("📭 No category score data available yet.")
        else:
            st.info("ℹ️ Missing category/category_score columns.")

    with chart_col2:
        if "polarity" in results_df.columns and not results_df["polarity"].empty:
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon pie">🥧</div>
                        <div class="result-card-title">Polarity Distribution</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            polarity_share = (
                results_df["polarity"].fillna("Unknown").value_counts(normalize=True).mul(100)
            )
            
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            # Custom colors for polarity - vibrant for light theme
            color_map = {
                'positive': '#10b981',
                'negative': '#ef4444', 
                'neutral': '#3b82f6',
                'Unknown': '#9ca3af'
            }
            colors = [color_map.get(str(p).lower(), '#667eea') for p in polarity_share.index]
            
            wedges, texts, autotexts = ax.pie(
                polarity_share.values.tolist(),
                labels=polarity_share.index.tolist(),
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.7, edgecolor='#ffffff', linewidth=3),
                textprops=dict(color='#1a202c', fontsize=10),
            )
            
            for autotext in autotexts:
                autotext.set_color('#1a202c')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ===== STATISTICS TABLE =====
    st.markdown(
        """
        <div class="result-card" style="margin-top: 24px;">
            <div class="result-card-header">
                <div class="result-card-icon chart">📊</div>
                <div class="result-card-title">Category & Polarity Statistics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "category" in results_df.columns:
        # Tạo bảng thống kê kết hợp
        category_stats = results_df.groupby("category").agg(
            Total_Clauses=("category", "count"),
        ).reset_index()
        
        # Thêm cột phần trăm
        category_stats["Percentage"] = (category_stats["Total_Clauses"] / category_stats["Total_Clauses"].sum() * 100).round(1)
        category_stats["Percentage"] = category_stats["Percentage"].astype(str) + "%"
        
        # Thêm thống kê polarity nếu có
        if "polarity" in results_df.columns:
            # Normalize polarity to lowercase để xử lý case-insensitive
            df_temp = results_df.copy()
            df_temp["polarity_lower"] = df_temp["polarity"].astype(str).str.lower().str.strip()
            polarity_counts = df_temp.groupby(["category", "polarity_lower"]).size().unstack(fill_value=0)
            
            for pol in ["positive", "neutral", "negative"]:
                col_name = pol.capitalize()
                if pol in polarity_counts.columns:
                    category_stats[col_name] = category_stats["category"].map(polarity_counts[pol]).fillna(0).astype(int)
                else:
                    category_stats[col_name] = 0
        
        # Thêm average score nếu có
        if "category_score" in results_df.columns:
            avg_score_by_category = results_df.groupby("category")["category_score"].mean().round(3)
            category_stats["Avg Score"] = category_stats["category"].map(avg_score_by_category)
        
        # Rename và sắp xếp
        category_stats = category_stats.rename(columns={"category": "Category", "Total_Clauses": "Total"})
        category_stats = category_stats.sort_values("Total", ascending=False)
        
        # Thêm hàng tổng (Summary row)
        total_row = {
            "Category": "📊 TOTAL",
            "Total": category_stats["Total"].sum(),
            "Percentage": "100%",
        }
        if "Positive" in category_stats.columns:
            total_row["Positive"] = category_stats["Positive"].sum()
        if "Neutral" in category_stats.columns:
            total_row["Neutral"] = category_stats["Neutral"].sum()
        if "Negative" in category_stats.columns:
            total_row["Negative"] = category_stats["Negative"].sum()
        if "Avg Score" in category_stats.columns:
            total_row["Avg Score"] = results_df["category_score"].mean().round(3) if "category_score" in results_df.columns else None
        
        category_stats = pd.concat([category_stats, pd.DataFrame([total_row])], ignore_index=True)
        
        st.dataframe(category_stats, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("ℹ️ No category data available.")

    # Detailed results table
    st.markdown(
        """
        <div class="result-card" style="margin-top: 24px;">
            <div class="result-card-header">
                <div class="result-card-icon table">📋</div>
                <div class="result-card-title">Detailed Analysis Results</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(results_df, use_container_width=True, height=400)
    
    # Download button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            "📥 Download Results as CSV",
            results_df.to_csv(index=False).encode("utf-8"),
            "absa_results.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 6️⃣ TAB: INPUT REVIEW & BATCH FILE
# =============================================================================
tab1, tab2 = st.tabs(["✍️ Single Review", "📁 Batch Analysis"])

# --- TAB 1: SINGLE REVIEW ---
with tab1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Enter your review",
        placeholder="Paste a customer review here to analyze its sentiment across different aspects...\n\nExample: 'The hotel room was spacious and clean, but the breakfast was disappointing. Staff were friendly and helpful.'",
        height=200,
        label_visibility="collapsed",
    )

    col_spacer, col_analyze, col_clear = st.columns([5, 1.5, 1])
    with col_analyze:
        analyze = st.button("🔍 Analyze", use_container_width=True)
    with col_clear:
        clear = st.button("🗑️ Clear", use_container_width=True, type="secondary")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.rerun()

    if analyze and text_input.strip():
        if not ensure_qwen_model_ready(models, announce=True):
            st.error("❌ Qwen model is not ready. Please check the configuration / pipeline.")
        else:
            with st.spinner("🔄 Analyzing review with AI..."):
                try:
                    df = absa_pipeline.run_full_pipeline(text_input, models)
                except Exception as e:
                    st.error(f"Error while running pipeline: {e}")
                    st.stop()

            if df.empty:
                st.warning("🔍 No aspects found in this review. Try a more detailed review.")
            else:
                render_dashboard(df)

# --- TAB 2: BATCH REVIEWS ---
with tab2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="text-align: center; padding: 16px 0;">
            <div style="font-size: 40px; margin-bottom: 12px;">📄</div>
            <div style="color: #a0a0b8; font-size: 15px; margin-bottom: 8px;">
                Upload a <code style="background: rgba(102, 126, 234, 0.2); padding: 2px 8px; border-radius: 4px; color: #a5b4fc;">.txt</code> file where each line is one review
            </div>
            <div style="color: #6b6b80; font-size: 13px;">
                Supports batch processing of multiple reviews at once
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt"], label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_batch = st.button("🚀 Run Batch Analysis", use_container_width=True)
    
    if run_batch:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a text file first.")
        else:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                reviews = [line.strip() for line in stringio.readlines() if line.strip()]
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            if not reviews:
                st.warning("📭 File is empty or invalid.")
            else:
                all_results = []
                st.info(f"🔄 Processing {len(reviews)} reviews...")
                progress = st.progress(0.0, text="Initializing...")
                qwen_failed = False

                for i, review in enumerate(reviews):
                    try:
                        if not ensure_qwen_model_ready(models, announce=(i == 0)):
                            st.error("❌ Qwen model is not ready for batch processing. Stopping.")
                            qwen_failed = True
                            break
                        
                        progress.progress((i + 1) / len(reviews), text=f"Analyzing review {i+1} of {len(reviews)}...")
                        df = absa_pipeline.run_full_pipeline(review, models)
                        
                        if not df.empty:
                            df["review_line"] = i + 1
                            df["review_text"] = review
                            all_results.append(df)
                    except Exception as e:
                        st.error(f"Error at line {i+1}: {e}")

                progress.empty()

                if qwen_failed:
                    st.info("🔧 Please check the model configuration and try again.")
                elif not all_results:
                    st.warning("📭 No aspects found in any review.")
                else:
                    final_df = pd.concat(all_results, ignore_index=True)
                    st.success(f"✅ Analysis complete! Found {len(final_df)} aspects across {len(reviews)} reviews.")
                    render_dashboard(final_df)

# Đóng div main-container
st.markdown("</div>", unsafe_allow_html=True)
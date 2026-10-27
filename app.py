"""
Stock Market Return Statistical Analyzer
A premium, financial-terminal-grade analytics and statistical dashboard
designed for academic quantitative research and market analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import joblib
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Stock Market Return Statistical Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM REFINED LIGHT THEME STYLING
# ==========================================
# Theme Tokens:
# Background: #F6F8FC
# Main cards: #FFFFFF
# Primary text: #17243A
# Secondary text: #69778C
# Accent blue: #315CEB
# Positive returns: #16856B
# Negative returns: #D94F59
# Borders: #E6EAF1

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        background-color: #F6F8FC !important;
        color: #17243A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #17243A !important;
        letter-spacing: -0.02em;
    }

    /* Refined Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E6EAF1 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Project Brand Badge in Sidebar */
    .brand-container {
        padding-bottom: 14px;
        margin-bottom: 16px;
        border-bottom: 1px solid #E6EAF1;
    }
    .brand-mark {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #315CEB;
        background: rgba(49, 92, 235, 0.08);
        border: 1px solid rgba(49, 92, 235, 0.18);
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #17243A;
        margin: 0;
        line-height: 1.3;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #69778C;
        margin-top: 3px;
        font-weight: 500;
    }
    
    /* Sidebar Navigation item styling */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-bottom: 2px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #17243A !important;
        transition: all 0.12s ease-in-out !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #F6F8FC !important;
    }
    
    /* Main App Header */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(23, 36, 58, 0.03);
    }
    .header-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 12px;
    }
    .identity-mark {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #315CEB;
        background: rgba(49, 92, 235, 0.08);
        border: 1px solid rgba(49, 92, 235, 0.18);
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .app-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        color: #17243A;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #69778C;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Section Headings */
    .page-header {
        margin-bottom: 20px;
    }
    .page-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 3px;
        letter-spacing: -0.02em;
    }
    .page-desc {
        font-size: 0.9rem;
        color: #69778C;
        margin-bottom: 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(23, 36, 58, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: #69778C;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #17243A;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 8px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 5px;
    }
    
    /* Badges */
    .badge-pos {
        color: #16856B;
        background-color: #E8F5F1;
        border: 1px solid #B2E2D4;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-neg {
        color: #D94F59;
        background-color: #FDF0F1;
        border: 1px solid #F5C2C6;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-neutral {
        color: #69778C;
        background-color: #F1F4F9;
        border: 1px solid #E6EAF1;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.76rem;
        font-weight: 500;
    }
    .badge-accent {
        color: #315CEB;
        background-color: rgba(49, 92, 235, 0.08);
        border: 1px solid rgba(49, 92, 235, 0.2);
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    
    /* Standard Content Cards */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(23, 36, 58, 0.02);
        margin-bottom: 16px;
    }
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }
    
    /* Key Observations Card */
    .observation-card {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-left: 3.5px solid #315CEB;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(23, 36, 58, 0.02);
    }
    .observation-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .observation-item {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 6px;
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .observation-bullet {
        color: #315CEB;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Academic Disclaimer Card */
    .disclaimer-card {
        background: #FAFBFD;
        border: 1px solid #E6EAF1;
        border-left: 3.5px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 24px;
        color: #475569;
        font-size: 0.84rem;
        line-height: 1.55;
    }
    
    /* Professional Comparison HTML Table */
    .table-container {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-radius: 8px;
        overflow-x: auto;
        box-shadow: 0 1px 2px rgba(23, 36, 58, 0.02);
    }
    .fin-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        text-align: left;
    }
    .fin-table th {
        background-color: #FAFBFD;
        color: #69778C;
        font-weight: 600;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 14px;
        border-bottom: 1px solid #E6EAF1;
    }
    .fin-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F1F4F9;
        color: #17243A;
    }
    .fin-table tr:last-child td {
        border-bottom: none;
    }
    .fin-table tr:hover td {
        background-color: #F8FAFD;
    }
    .fin-table tr.active-row td {
        background-color: rgba(49, 92, 235, 0.04);
        font-weight: 600;
    }
    
    /* Streamlit Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E6EAF1;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 16px;
        color: #69778C;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #315CEB !important;
        border-bottom-color: #315CEB !important;
    }
    
    /* Streamlit DataFrame Override */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E6EAF1;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Control Strip Container */
    .control-strip {
        background: #FFFFFF;
        border: 1px solid #E6EAF1;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PLOTLY REFINED LIGHT THEME BUILDER
# ==========================================
def create_clean_layout(
    title="",
    xaxis_title="",
    yaxis_title="",
    height=380,
    hovermode="x unified",
    show_legend=True
):
    """Creates a consistent, publication-ready light theme layout for Plotly figures."""
    layout = go.Layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#17243A", size=11),
        title=dict(
            text=title,
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#17243A", weight=700),
            x=0.01,
            y=0.96
        ) if title else None,
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11, color="#69778C", weight=500)),
            gridcolor="#F0F4F8",
            gridwidth=1,
            zerolinecolor="#E6EAF1",
            zerolinewidth=1,
            tickfont=dict(color="#69778C", size=10),
            showline=True,
            linecolor="#E6EAF1",
            linewidth=1
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color="#69778C", weight=500)),
            gridcolor="#F0F4F8",
            gridwidth=1,
            zerolinecolor="#E6EAF1",
            zerolinewidth=1,
            tickfont=dict(color="#69778C", size=10),
            showline=True,
            linecolor="#E6EAF1",
            linewidth=1
        ),
        margin=dict(l=48, r=20, t=38 if title else 20, b=38),
        height=height,
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor="#17243A",
            bordercolor="#17243A",
            font=dict(color="#FFFFFF", family="Inter, sans-serif", size=11)
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10.5, color="#69778C"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E6EAF1",
            borderwidth=1
        ) if show_legend else None
    )
    return layout


# ==========================================
# DATA & MODEL LOADING WITH CACHING
# ==========================================
APP_DIR = Path(__file__).resolve().parent

def resolve_file(rel_path: str):
    """Finds data or model files across potential workspace directories."""
    candidates = [
        APP_DIR / rel_path,
        APP_DIR / "stock_analyzer_data" / Path(rel_path).name,
        APP_DIR / "models" / Path(rel_path).name,
        APP_DIR / "scratch" / Path(rel_path).name,
        Path.cwd() / rel_path,
        Path.cwd() / "stock_analyzer_data" / Path(rel_path).name,
        Path.cwd() / "models" / Path(rel_path).name,
        Path.cwd() / "scratch" / Path(rel_path).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

@st.cache_data(show_spinner=False)
def load_data():
    """Loads and standardizes CSV data files."""
    data = {}
    errors = []

    # 1. Close prices
    p_close = resolve_file("stock_analyzer_data/stock_close_prices.csv")
    if p_close:
        try:
            df = pd.read_csv(p_close)
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df = df.set_index(date_col)
            data["close_prices"] = df
        except Exception as e:
            errors.append(f"Failed parsing stock_close_prices.csv: {e}")
    else:
        errors.append("stock_close_prices.csv not found.")

    # 2. Daily returns
    p_ret = resolve_file("stock_analyzer_data/daily_stock_returns.csv")
    if p_ret:
        try:
            df = pd.read_csv(p_ret)
            date_col = "index" if "index" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df = df.set_index(date_col)
            data["returns"] = df
        except Exception as e:
            errors.append(f"Failed parsing daily_stock_returns.csv: {e}")
    else:
        errors.append("daily_stock_returns.csv not found.")

    # 3. Modeling dataset
    p_model = resolve_file("stock_analyzer_data/modeling_dataset.csv")
    if p_model:
        try:
            df = pd.read_csv(p_model)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                df = df.set_index("Date")
            data["modeling_dataset"] = df
        except Exception as e:
            errors.append(f"Failed parsing modeling_dataset.csv: {e}")

    # 4. Model comparison CSVs
    p_reg = resolve_file("stock_analyzer_data/regression_model_comparison.csv")
    if p_reg:
        data["regression_comparison"] = pd.read_csv(p_reg)

    p_clf = resolve_file("stock_analyzer_data/classification_model_comparison.csv")
    if p_clf:
        data["classification_comparison"] = pd.read_csv(p_clf)

    p_cm = resolve_file("stock_analyzer_data/classification_confusion_matrix.csv")
    if p_cm:
        data["confusion_matrix"] = pd.read_csv(p_cm, index_col=0)

    return data, errors

@st.cache_resource(show_spinner=False)
def load_models():
    """Loads fitted scikit-learn model pipelines with cross-version attribute compatibility."""
    models = {}
    errors = []

    p_reg = resolve_file("models/best_return_regression_model.pkl")
    p_clf = resolve_file("models/best_direction_classification_model.pkl")

    if p_reg:
        try:
            reg = joblib.load(p_reg)
            imp = getattr(reg, "named_steps", {}).get("imputer")
            if imp is not None and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = getattr(imp, "_fit_dtype", float)
            models["regression"] = reg
        except Exception as e:
            errors.append(f"Regression model error: {e}")

    if p_clf:
        try:
            clf = joblib.load(p_clf)
            imp = getattr(clf, "named_steps", {}).get("imputer")
            if imp is not None and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = getattr(imp, "_fit_dtype", float)
            models["classification"] = clf
        except Exception as e:
            errors.append(f"Classification model error: {e}")

    return models, errors

def calculate_descriptive_stats(series: pd.Series) -> pd.Series:
    """Calculates statistical moments and distribution metrics."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    
    q1 = float(clean.quantile(0.25))
    q2 = float(clean.quantile(0.50))
    q3 = float(clean.quantile(0.75))
    
    mode_vals = clean.round(4).mode()
    mode_val = float(mode_vals.iloc[0]) if not mode_vals.empty else q2
    
    return pd.Series({
        "Count": int(clean.count()),
        "Mean (%)": float(clean.mean()),
        "Median (%)": q2,
        "Mode (%)": mode_val,
        "Variance (%²)": float(clean.var(ddof=1)),
        "Std Deviation (%)": float(clean.std(ddof=1)),
        "Minimum (%)": float(clean.min()),
        "Q1 (25th %)": q1,
        "Q2 (Median %)": q2,
        "Q3 (75th %)": q3,
        "Maximum (%)": float(clean.max()),
        "IQR (%)": q3 - q1,
        "Skewness": float(clean.skew()),
        "Kurtosis": float(clean.kurtosis())
    })


# ==========================================
# DATA INITIALIZATION & VALIDATION
# ==========================================
data_dict, data_errors = load_data()
model_dict, model_errors = load_models()

if "returns" not in data_dict or "close_prices" not in data_dict:
    st.error("Essential market data files could not be loaded. Please ensure dataset CSVs are present.")
    st.stop()

df_prices = data_dict["close_prices"]
df_returns = data_dict["returns"]
df_model_data = data_dict.get("modeling_dataset", None)

all_tickers = [c for c in df_returns.columns if c not in ["Date", "index"]]
BENCHMARK = "^GSPC"
stock_tickers = [t for t in all_tickers if t != BENCHMARK]
DEFAULT_TICKER = "AAPL" if "AAPL" in stock_tickers else stock_tickers[0]


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = DEFAULT_TICKER
if "date_preset" not in st.session_state:
    st.session_state["date_preset"] = "Full 5 Years"
if "use_benchmark" not in st.session_state:
    st.session_state["use_benchmark"] = BENCHMARK in df_returns.columns


# ==========================================
# SIDEBAR CONTROLS & REFINED NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-mark">FINANCIAL ANALYTICS</div>
        <div class="brand-title">Stock Return Analyzer</div>
        <div class="brand-subtitle">Academic Research Terminal</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Primary Equity Selection
    sb_stock = st.selectbox(
        "Target Stock",
        options=stock_tickers,
        index=stock_tickers.index(st.session_state["selected_stock"]) if st.session_state["selected_stock"] in stock_tickers else 0,
        help="Select the equity asset to analyze."
    )
    st.session_state["selected_stock"] = sb_stock
    
    # 2. Benchmark Inclusion
    has_benchmark = BENCHMARK in df_returns.columns
    sb_benchmark = st.checkbox(
        "Compare with S&P 500 (^GSPC)",
        value=st.session_state["use_benchmark"],
        disabled=not has_benchmark,
        help="Include S&P 500 benchmark in charts and comparisons."
    )
    st.session_state["use_benchmark"] = sb_benchmark
    
    st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E6EAF1;'>", unsafe_allow_html=True)
    
    # 3. Date Filter
    min_date = df_returns.index.min().to_pydatetime()
    max_date = df_returns.index.max().to_pydatetime()
    
    date_preset = st.radio(
        "Date Window",
        ["Full 5 Years", "Past 3 Years", "Past 1 Year", "Custom Range"],
        index=["Full 5 Years", "Past 3 Years", "Past 1 Year", "Custom Range"].index(st.session_state["date_preset"])
    )
    st.session_state["date_preset"] = date_preset
    
    if date_preset == "Past 1 Year":
        filter_start = max(min_date, max_date - pd.Timedelta(days=365))
        filter_end = max_date
    elif date_preset == "Past 3 Years":
        filter_start = max(min_date, max_date - pd.Timedelta(days=365 * 3))
        filter_end = max_date
    elif date_preset == "Custom Range":
        filter_range = st.date_input(
            "Select Dates",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        if isinstance(filter_range, (tuple, list)) and len(filter_range) == 2:
            filter_start = pd.to_datetime(filter_range[0])
            filter_end = pd.to_datetime(filter_range[1])
        else:
            filter_start = min_date
            filter_end = max_date
    else:
        filter_start = min_date
        filter_end = max_date
        
    ts_start = pd.to_datetime(filter_start)
    ts_end = pd.to_datetime(filter_end)
    
    st.markdown("<hr style='margin: 14px 0; border: none; border-top: 1px solid #E6EAF1;'>", unsafe_allow_html=True)
    
    # 4. Navigation Menu
    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Historical Trends",
            "Statistics",
            "Return Distribution",
            "Sampling & CLT",
            "Hypothesis Testing",
            "Correlation Analysis",
            "Model Performance",
            "Predictions"
        ],
        index=0
    )

selected_stock = st.session_state["selected_stock"]
use_benchmark = st.session_state["use_benchmark"]

# Slice active data according to selected date filter
active_returns = df_returns.loc[(df_returns.index >= ts_start) & (df_returns.index <= ts_end)].copy()
active_prices = df_prices.loc[(df_prices.index >= ts_start) & (df_prices.index <= ts_end)].copy()

stock_return_series = active_returns[selected_stock].dropna()
stock_price_series = active_prices[selected_stock].dropna() if selected_stock in active_prices.columns else pd.Series(dtype=float)
benchmark_return_series = active_returns[BENCHMARK].dropna() if (use_benchmark and BENCHMARK in active_returns.columns) else None
benchmark_price_series = active_prices[BENCHMARK].dropna() if (use_benchmark and BENCHMARK in active_prices.columns) else None

latest_avail_date_str = df_prices.index.max().strftime("%B %d, %Y")


# ==========================================
# 1. OVERVIEW PAGE (HERO / LANDING)
# ==========================================
if nav_selection == "Overview":
    # A. Header Area
    st.markdown(f"""
    <div class="app-header">
        <div class="header-top">
            <div>
                <div class="identity-mark">PORTFOLIO & RISK RESEARCH</div>
                <div class="app-title">Stock Market Return Statistical Analyzer</div>
                <div class="app-subtitle">Explore market performance, risk, and statistical patterns.</div>
            </div>
            <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                <span class="badge-neutral" style="font-size: 0.78rem; font-weight: 600;">Data as of {latest_avail_date_str}</span>
                <span style="font-size: 0.76rem; color: #69778C;">Active: <strong>{selected_stock}</strong> • {ts_start.strftime('%b %Y')} – {ts_end.strftime('%b %Y')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty or stock_price_series.empty:
        st.warning("No price data available for the selected dates.")
    else:
        # B. Key Metrics Calculations
        p_start = float(stock_price_series.iloc[0])
        p_end = float(stock_price_series.iloc[-1])
        period_return = ((p_end - p_start) / p_start) * 100
        
        # Dataset inception calculation (full dataset start to end)
        full_price_series = df_prices[selected_stock].dropna()
        p_inception_start = float(full_price_series.iloc[0])
        inception_return = ((p_end - p_inception_start) / p_inception_start) * 100
        
        avg_daily_return = float(stock_return_series.mean())
        daily_volatility = float(stock_return_series.std(ddof=1))
        annualized_return = avg_daily_return * 252
        annualized_volatility = daily_volatility * np.sqrt(252)
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0
        
        # Benchmark calculations
        bm_period_return = 0.0
        if benchmark_price_series is not None and not benchmark_price_series.empty:
            bm_start = float(benchmark_price_series.iloc[0])
            bm_end = float(benchmark_price_series.iloc[-1])
            bm_period_return = ((bm_end - bm_start) / bm_start) * 100
            
        # Drawdown calculation
        rolling_max = stock_price_series.cummax()
        drawdowns = (stock_price_series - rolling_max) / rolling_max * 100
        max_drawdown = float(drawdowns.min())

        # B. KPI Strip (4 compact cards)
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Close</div>
                <div class="metric-value">${p_end:,.2f}</div>
                <div class="metric-sub">
                    <span class="badge-neutral">{selected_stock}</span>
                    <span style="color: #69778C; font-size: 0.78rem;">Period Base: ${p_start:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            ret_sign = "+" if period_return >= 0 else ""
            ret_badge = "badge-pos" if period_return >= 0 else "badge-neg"
            inc_sign = "+" if inception_return >= 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value" style="color: {'#16856B' if period_return >= 0 else '#D94F59'};">{ret_sign}{period_return:.2f}%</div>
                <div class="metric-sub">
                    <span class="{ret_badge}">{ret_sign}{period_return:.1f}% Selected</span>
                    <span class="badge-neutral">Inception: {inc_sign}{inception_return:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            avg_sign = "+" if avg_daily_return >= 0 else ""
            avg_badge = "badge-pos" if avg_daily_return >= 0 else "badge-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Daily Return</div>
                <div class="metric-value">{avg_sign}{avg_daily_return:.3f}%</div>
                <div class="metric-sub">
                    <span class="{avg_badge}">Annualized: {avg_sign}{annualized_return:.1f}% p.a.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Volatility</div>
                <div class="metric-value">{daily_volatility:.2f}%</div>
                <div class="metric-sub">
                    <span class="badge-neutral">Annualized: {annualized_volatility:.1f}% p.a.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # C. Main Performance Visualization (Rebased to 100)
        norm_stock = (stock_price_series / p_start) * 100
        cum_gain_stock = norm_stock - 100
        
        fig_main = go.Figure()
        
        fig_main.add_trace(go.Scatter(
            x=norm_stock.index,
            y=norm_stock.values,
            mode='lines',
            name=f"{selected_stock} ({ret_sign}{period_return:.1f}%)",
            line=dict(color='#315CEB', width=2.4),
            customdata=cum_gain_stock.values,
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Normalized: <b>%{y:.2f}</b><br>Period Return: <b>%{customdata:+.2f}%</b><extra></extra>"
        ))
        
        if benchmark_price_series is not None and not benchmark_price_series.empty:
            bm_start_val = float(benchmark_price_series.iloc[0])
            norm_bm = (benchmark_price_series / bm_start_val) * 100
            cum_gain_bm = norm_bm - 100
            bm_sign = "+" if bm_period_return >= 0 else ""
            
            fig_main.add_trace(go.Scatter(
                x=norm_bm.index,
                y=norm_bm.values,
                mode='lines',
                name=f"S&P 500 (^GSPC) ({bm_sign}{bm_period_return:.1f}%)",
                line=dict(color='#69778C', width=1.8, dash='dash'),
                customdata=cum_gain_bm.values,
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>S&P 500 Normalized: <b>%{y:.2f}</b><br>Return: <b>%{customdata:+.2f}%</b><extra></extra>"
            ))
            
        # Rebase reference line at 100
        fig_main.add_hline(
            y=100,
            line_dash="dot",
            line_color="#CBD5E1",
            line_width=1.2,
            annotation_text="Base 100",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#69778C")
        )
        
        main_layout = create_clean_layout(
            title=f"Normalized Growth: {selected_stock} vs. S&P 500 Benchmark (Rebased to 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Index (Start = 100)",
            height=410
        )
        
        # Add range selector buttons [1M, 3M, 6M, 1Y, All]
        main_layout.xaxis.update(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                font=dict(family="Inter, sans-serif", size=10, color="#17243A"),
                bgcolor="#FAFBFD",
                activecolor="#E8F0FE",
                bordercolor="#E6EAF1",
                borderwidth=1
            )
        )
        fig_main.update_layout(main_layout)
        st.plotly_chart(fig_main, use_container_width=True)
        st.caption("*Both asset series are rebased to 100.0 at the beginning of the selected period to make relative capital performance directly comparable.")
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # D. Secondary Chart Row (Two-column layout)
        col_sec1, col_sec2 = st.columns(2)
        
        with col_sec1:
            ret_colors = ['#16856B' if v >= 0 else '#D94F59' for v in stock_return_series.values]
            fig_daily = go.Figure(go.Bar(
                x=stock_return_series.index,
                y=stock_return_series.values,
                marker_color=ret_colors,
                name="Daily Return",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Daily Return: <b>%{y:+.2f}%</b><extra></extra>"
            ))
            fig_daily.add_hline(y=0, line_color="#69778C", line_width=1.2)
            fig_daily.update_layout(create_clean_layout(
                title=f"{selected_stock} Daily Percentage Returns",
                xaxis_title="Date",
                yaxis_title="Daily Return (%)",
                height=350,
                show_legend=False
            ))
            st.plotly_chart(fig_daily, use_container_width=True)
            
        with col_sec2:
            # Risk vs Return scatter for all stocks
            scatter_tickers = [t for t in stock_tickers if t in active_returns.columns]
            if use_benchmark and BENCHMARK in active_returns.columns:
                scatter_tickers.append(BENCHMARK)
                
            vol_vals = []
            ret_vals = []
            symbols = []
            colors = []
            sizes = []
            labels = []
            
            for t in scatter_tickers:
                s_ret = active_returns[t].dropna()
                if not s_ret.empty:
                    ann_v = float(s_ret.std(ddof=1) * np.sqrt(252))
                    ann_r = float(s_ret.mean() * 252)
                    vol_vals.append(ann_v)
                    ret_vals.append(ann_r)
                    labels.append(t)
                    
                    if t == selected_stock:
                        symbols.append('circle')
                        colors.append('#315CEB')
                        sizes.append(14)
                    elif t == BENCHMARK:
                        symbols.append('diamond')
                        colors.append('#17243A')
                        sizes.append(12)
                    else:
                        symbols.append('circle')
                        colors.append('#69778C')
                        sizes.append(10)
                        
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(
                x=vol_vals,
                y=ret_vals,
                mode='markers+text',
                text=labels,
                textposition="top center",
                textfont=dict(family="Plus Jakarta Sans, sans-serif", size=10, color="#17243A"),
                marker=dict(
                    symbol=symbols,
                    color=colors,
                    size=sizes,
                    line=dict(color="#17243A", width=1.2)
                ),
                hovertemplate="<b>%{text}</b><br>Annualized Volatility: %{x:.1f}%<br>Annualized Return: %{y:+.1f}%<extra></extra>"
            ))
            fig_rr.add_hline(y=0, line_dash="dash", line_color="#E6EAF1", line_width=1)
            fig_rr.update_layout(create_clean_layout(
                title="Risk vs. Return Profile (Annualized)",
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                height=350,
                show_legend=False
            ))
            st.plotly_chart(fig_rr, use_container_width=True)
            st.caption("*Annualized assuming 252 trading sessions per year: Return = μ × 252, Volatility = σ × √252.")
            
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # E. Stock Comparison Table
        st.markdown("### Stock & Benchmark Comparative Overview")
        
        comp_rows = []
        all_eval_tickers = [t for t in stock_tickers if t in active_returns.columns and t in active_prices.columns]
        if BENCHMARK in active_returns.columns and BENCHMARK in active_prices.columns:
            all_eval_tickers.append(BENCHMARK)
            
        for t in all_eval_tickers:
            r_s = active_returns[t].dropna()
            p_s = active_prices[t].dropna()
            if not r_s.empty and not p_s.empty:
                c_ret = ((p_s.iloc[-1] - p_s.iloc[0]) / p_s.iloc[0]) * 100
                d_ret = float(r_s.mean())
                d_vol = float(r_s.std(ddof=1))
                a_vol = d_vol * np.sqrt(252)
                a_ret = d_ret * 252
                sh = (a_ret / a_vol) if a_vol > 0 else 0.0
                comp_rows.append({
                    "ticker": t,
                    "is_active": t == selected_stock,
                    "latest_price": p_s.iloc[-1],
                    "period_ret": c_ret,
                    "avg_daily_ret": d_ret,
                    "ann_vol": a_vol,
                    "sharpe": sh
                })
                
        # Render clean HTML financial table
        html_rows = ""
        for r in comp_rows:
            active_cls = ' class="active-row"' if r['is_active'] else ''
            p_sign = "+" if r['period_ret'] >= 0 else ""
            p_cls = "badge-pos" if r['period_ret'] >= 0 else "badge-neg"
            d_sign = "+" if r['avg_daily_ret'] >= 0 else ""
            d_cls = "badge-pos" if r['avg_daily_ret'] >= 0 else "badge-neg"
            
            ticker_display = f"<strong>{r['ticker']}</strong>"
            if r['is_active']:
                ticker_display += ' <span class="badge-accent" style="margin-left: 4px;">Selected</span>'
            elif r['ticker'] == BENCHMARK:
                ticker_display += ' <span class="badge-neutral" style="margin-left: 4px;">Benchmark</span>'
                
            html_rows += f"""
            <tr{active_cls}>
                <td>{ticker_display}</td>
                <td style="text-align: right; font-variant-numeric: tabular-nums;">${r['latest_price']:,.2f}</td>
                <td style="text-align: right;"><span class="{p_cls}">{p_sign}{r['period_ret']:.2f}%</span></td>
                <td style="text-align: right;"><span class="{d_cls}">{d_sign}{r['avg_daily_ret']:.3f}%</span></td>
                <td style="text-align: right; font-variant-numeric: tabular-nums;">{r['ann_vol']:.2f}%</td>
                <td style="text-align: right; font-variant-numeric: tabular-nums;">{r['sharpe']:.2f}</td>
            </tr>
            """
            
        st.markdown(f"""
        <div class="table-container">
            <table class="fin-table">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th style="text-align: right;">Latest Close</th>
                        <th style="text-align: right;">Period Return</th>
                        <th style="text-align: right;">Avg Daily Return</th>
                        <th style="text-align: right;">Annualized Volatility</th>
                        <th style="text-align: right;">Sharpe Ratio (Rf=0%)</th>
                    </tr>
                </thead>
                <tbody>
                    {html_rows}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # F. Key Observations (Up to three short statements generated from data)
        # 1. Top return asset
        top_ret_asset = max(comp_rows, key=lambda x: x['period_ret']) if comp_rows else None
        # 2. Highest annualized volatility
        top_vol_asset = max(comp_rows, key=lambda x: x['ann_vol']) if comp_rows else None
        lowest_vol_asset = min(comp_rows, key=lambda x: x['ann_vol']) if comp_rows else None
        # 3. Relative outperformance vs S&P 500
        bench_row = next((r for r in comp_rows if r['ticker'] == BENCHMARK), None)
        active_row = next((r for r in comp_rows if r['ticker'] == selected_stock), None)
        
        obs_list = []
        if top_ret_asset:
            t_sign = "+" if top_ret_asset['period_ret'] >= 0 else ""
            obs_list.append(
                f"<strong>Highest Period Return:</strong> <strong>{top_ret_asset['ticker']}</strong> delivered the highest return of "
                f"<strong>{t_sign}{top_ret_asset['period_ret']:.1f}%</strong> over the selected observation window."
            )
            
        if top_vol_asset and lowest_vol_asset:
            obs_list.append(
                f"<strong>Volatility Spread:</strong> <strong>{top_vol_asset['ticker']}</strong> exhibited the highest annualized volatility at "
                f"<strong>{top_vol_asset['ann_vol']:.1f}%</strong>, compared to the lowest observed volatility of "
                f"<strong>{lowest_vol_asset['ann_vol']:.1f}%</strong> ({lowest_vol_asset['ticker']})."
            )
            
        if bench_row and active_row:
            diff = active_row['period_ret'] - bench_row['period_ret']
            direction = "outperformed" if diff >= 0 else "underperformed"
            diff_sign = "+" if diff >= 0 else ""
            obs_list.append(
                f"<strong>Benchmark Comparison:</strong> <strong>{selected_stock}</strong> {direction} the S&P 500 benchmark by "
                f"<strong>{diff_sign}{diff:.1f} percentage points</strong> ({active_row['period_ret']:+.1f}% vs. {bench_row['period_ret']:+.1f}%)."
            )
            
        obs_html = "".join([f'<div class="observation-item"><span class="observation-bullet">•</span><div>{item}</div></div>' for item in obs_list[:3]])
        
        st.markdown(f"""
        <div class="observation-card">
            <div class="observation-title">📌 Key Analytical Observations</div>
            {obs_html}
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 2. HISTORICAL TRENDS
# ==========================================
elif nav_selection == "Historical Trends":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Historical Trends</div>
        <div class="page-desc">Track closing prices, technical moving averages, and cumulative portfolio growth.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_price_series.empty or stock_return_series.empty:
        st.warning("Insufficient data available for the active selection.")
    else:
        tab_t1, tab_t2, tab_t3 = st.tabs([
            "Price Action & Moving Averages",
            "Normalized Growth ($10,000)",
            "Daily Returns & Drawdowns"
        ])
        
        with tab_t1:
            ma20 = stock_price_series.rolling(20).mean()
            ma50 = stock_price_series.rolling(50).mean()
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=stock_price_series.index,
                y=stock_price_series.values,
                mode='lines',
                name=f"{selected_stock} Close",
                line=dict(color='#315CEB', width=2.2),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: <b>$%{y:,.2f}</b><extra></extra>"
            ))
            fig_p.add_trace(go.Scatter(
                x=ma20.index,
                y=ma20.values,
                mode='lines',
                name="20-Day MA",
                line=dict(color='#F59E0B', width=1.5, dash='dash'),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>20-Day MA: <b>$%{y:,.2f}</b><extra></extra>"
            ))
            fig_p.add_trace(go.Scatter(
                x=ma50.index,
                y=ma50.values,
                mode='lines',
                name="50-Day MA",
                line=dict(color='#16856B', width=1.5, dash='dot'),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>50-Day MA: <b>$%{y:,.2f}</b><extra></extra>"
            ))
            
            fig_p.update_layout(create_clean_layout(
                title=f"{selected_stock} Adjusted Closing Price with 20-Day and 50-Day Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($ USD)",
                height=420
            ))
            st.plotly_chart(fig_p, use_container_width=True)
            
            with st.expander("Technical Context: Moving Averages"):
                st.caption(
                    "A 20-day moving average reflects short-term market momentum, while a 50-day moving average smooths high-frequency noise "
                    "to reveal medium-term structural trends without introducing subjective bias."
                )
                
        with tab_t2:
            growth_stock = (1 + stock_return_series / 100).cumprod() * 10000
            
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=growth_stock.index,
                y=growth_stock.values,
                mode='lines',
                name=f"{selected_stock} Portfolio",
                line=dict(color='#315CEB', width=2.4),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Value: <b>$%{y:,.2f}</b><extra></extra>"
            ))
            
            if benchmark_return_series is not None and not benchmark_return_series.empty:
                growth_bm = (1 + benchmark_return_series / 100).cumprod() * 10000
                fig_g.add_trace(go.Scatter(
                    x=growth_bm.index,
                    y=growth_bm.values,
                    mode='lines',
                    name="S&P 500 (^GSPC)",
                    line=dict(color='#69778C', width=1.8, dash='dash'),
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>S&P 500: <b>$%{y:,.2f}</b><extra></extra>"
                ))
                
            fig_g.add_hline(
                y=10000,
                line_dash="dot",
                line_color="#CBD5E1",
                annotation_text="Initial $10,000",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#69778C")
            )
            
            fig_g.update_layout(create_clean_layout(
                title="Cumulative Growth of an Initial $10,000 Investment",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($ USD)",
                height=420
            ))
            st.plotly_chart(fig_g, use_container_width=True)
            
            final_val = float(growth_stock.iloc[-1])
            tot_gain = final_val - 10000
            st.caption(
                f"An initial $10,000 investment in **{selected_stock}** on {stock_return_series.index.min().strftime('%b %d, %Y')} "
                f"would be valued at **${final_val:,.2f}** ({'+' if tot_gain >= 0 else ''}${tot_gain:,.2f}) as of {stock_return_series.index.max().strftime('%b %d, %Y')}."
            )
            
        with tab_t3:
            col_b, col_d = st.columns(2)
            
            with col_b:
                bar_cols = ['#16856B' if v >= 0 else '#D94F59' for v in stock_return_series.values]
                fig_bar = go.Figure(go.Bar(
                    x=stock_return_series.index,
                    y=stock_return_series.values,
                    marker_color=bar_cols,
                    name="Daily Return",
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Return: <b>%{y:+.2f}%</b><extra></extra>"
                ))
                fig_bar.add_hline(y=0, line_color="#69778C", line_width=1.2)
                fig_bar.update_layout(create_clean_layout(
                    title=f"{selected_stock} Daily Percentage Return Fluctuations",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=360,
                    show_legend=False
                ))
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_d:
                rolling_peak = stock_price_series.cummax()
                dd_series = (stock_price_series - rolling_peak) / rolling_peak * 100
                
                fig_dd = go.Figure(go.Scatter(
                    x=dd_series.index,
                    y=dd_series.values,
                    mode='lines',
                    line=dict(color='#D94F59', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(217, 79, 89, 0.12)',
                    name="Drawdown",
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Drawdown: <b>%{y:.2f}%</b><extra></extra>"
                ))
                fig_dd.add_hline(y=0, line_color="#69778C", line_width=1)
                fig_dd.update_layout(create_clean_layout(
                    title=f"{selected_stock} Peak-to-Trough Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown from Peak (%)",
                    height=360,
                    show_legend=False
                ))
                st.plotly_chart(fig_dd, use_container_width=True)


# ==========================================
# 3. STATISTICS
# ==========================================
elif nav_selection == "Statistics":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Descriptive Statistics</div>
        <div class="page-desc">Quantify central tendency, dispersion, and distribution asymmetry for equity returns.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("No data points available for the selected dates.")
    else:
        stats_series = calculate_descriptive_stats(stock_return_series)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Central Tendency ({selected_stock})</div>
                <div style="font-size: 0.88rem; line-height: 1.85;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Mean Daily Return:</span>
                        <strong>{stats_series['Mean (%)']:+.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Median Daily Return:</span>
                        <strong>{stats_series['Median (%)']:+.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Sample Mode:</span>
                        <strong>{stats_series['Mode (%)']:+.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: #69778C;">Sample Size (n):</span>
                        <strong>{int(stats_series['Count']):,} sessions</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Spread & Dispersion ({selected_stock})</div>
                <div style="font-size: 0.88rem; line-height: 1.85;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Sample Std Dev (s):</span>
                        <strong>{stats_series['Std Deviation (%)']:.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Sample Variance (s²):</span>
                        <strong>{stats_series['Variance (%²)']:.4f}%²</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Interquartile Range (IQR):</span>
                        <strong>{stats_series['IQR (%)']:.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: #69778C;">Range [Min, Max]:</span>
                        <strong>[{stats_series['Minimum (%)']:.2f}%, {stats_series['Maximum (%)']:.2f}%]</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s3:
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Shape & Percentiles ({selected_stock})</div>
                <div style="font-size: 0.88rem; line-height: 1.85;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Fisher Skewness:</span>
                        <strong>{stats_series['Skewness']:+.4f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">Excess Kurtosis:</span>
                        <strong>{stats_series['Kurtosis']:.4f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                        <span style="color: #69778C;">25th Percentile (Q1):</span>
                        <strong>{stats_series['Q1 (25th %)']:+.4f}%</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: #69778C;">75th Percentile (Q3):</span>
                        <strong>{stats_series['Q3 (75th %)']:+.4f}%</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Visual Comparison: Mean Return vs Volatility Bar Chart across all available assets
        st.markdown("### Visual Comparison: Return vs. Volatility by Asset")
        
        comp_tickers = [t for t in stock_tickers if t in active_returns.columns]
        if use_benchmark and BENCHMARK in active_returns.columns:
            comp_tickers.append(BENCHMARK)
            
        means_list = []
        vols_list = []
        medians_list = []
        for t in comp_tickers:
            s_t = active_returns[t].dropna()
            means_list.append(float(s_t.mean()))
            vols_list.append(float(s_t.std(ddof=1)))
            medians_list.append(float(s_t.median()))
            
        fig_stat = go.Figure()
        fig_stat.add_trace(go.Bar(
            x=comp_tickers,
            y=means_list,
            name="Daily Mean Return (%)",
            marker_color='#315CEB'
        ))
        fig_stat.add_trace(go.Bar(
            x=comp_tickers,
            y=medians_list,
            name="Daily Median Return (%)",
            marker_color='#16856B'
        ))
        fig_stat.add_trace(go.Bar(
            x=comp_tickers,
            y=vols_list,
            name="Daily Volatility / Std Dev (%)",
            marker_color='#69778C'
        ))
        
        fig_stat.add_hline(y=0, line_color="#E6EAF1", line_width=1)
        fig_stat.update_layout(create_clean_layout(
            title="Comparison of Daily Mean, Median, and Volatility across Assets",
            xaxis_title="Asset",
            yaxis_title="Percentage (%)",
            height=370
        ))
        st.plotly_chart(fig_stat, use_container_width=True)
        
        # Cross-asset comparison table
        st.markdown("### Cross-Stock Descriptive Statistics Table")
        all_stats = {}
        for t in comp_tickers:
            s = active_returns[t].dropna()
            if not s.empty:
                all_stats[t] = calculate_descriptive_stats(s)
                
        df_all_stats = pd.DataFrame(all_stats).T
        st.dataframe(df_all_stats.round(4), use_container_width=True)
        
        with st.expander("Statistical Concept Reference"):
            st.markdown("""
            - **Skewness:** Measures the symmetry of the probability distribution. Positive skew indicates longer right tails (occasional large positive gains), while negative skew indicates downside tail risk.
            - **Excess Kurtosis:** Quantifies heavy-tail probability relative to a Gaussian distribution (0.0). Positive excess kurtosis (leptokurtosis) is typical of financial returns, implying extreme daily shocks occur far more frequently than predicted by normal models.
            - **IQR (Interquartile Range):** The difference between Q3 (75th percentile) and Q1 (25th percentile), covering the central 50% of observed return outcomes.
            """)


# ==========================================
# 4. RETURN DISTRIBUTION
# ==========================================
elif nav_selection == "Return Distribution":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Return Distribution</div>
        <div class="page-desc">Examine empirical return density, theoretical Gaussian models, and outlier dispersion.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("No data points available for the selected dates.")
    else:
        clean_ret = stock_return_series.to_numpy()
        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        
        tab_d1, tab_d2, tab_d3 = st.tabs([
            "Histogram & Normal Curve",
            "Box Plot & Outliers",
            "Empirical Rule vs. Chebyshev"
        ])
        
        with tab_d1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=clean_ret,
                histnorm='probability density',
                nbinsx=50,
                name="Empirical Returns",
                marker_color='#315CEB',
                opacity=0.65,
                hovertemplate="Bin Range: %{x}<br>Density: %{y:.4f}<extra></extra>"
            ))
            
            x_grid = np.linspace(clean_ret.min() - 0.5, clean_ret.max() + 0.5, 200)
            norm_pdf = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
            fig_h.add_trace(go.Scatter(
                x=x_grid,
                y=norm_pdf,
                mode='lines',
                name=f"Theoretical Normal N(μ={mu:.2f}%, s={sigma:.2f}%)",
                line=dict(color='#F59E0B', width=2.2),
                hovertemplate="Return: %{x:.2f}%<br>Normal Density: %{y:.4f}<extra></extra>"
            ))
            
            # Zero and Mean vertical reference lines
            fig_h.add_vline(x=0, line_color="#69778C", line_width=1.2, line_dash="solid")
            fig_h.add_vline(x=mu, line_color="#16856B", line_width=1.5, line_dash="dash", annotation_text=f"Mean ({mu:+.2f}%)", annotation_position="top right")
            
            fig_h.update_layout(create_clean_layout(
                title=f"{selected_stock} Daily Return Histogram with Fitted Gaussian Normal Curve",
                xaxis_title="Daily Return (%)",
                yaxis_title="Probability Density",
                height=420
            ))
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption("Notice the characteristic leptokurtic peak near 0% and extended tails compared to the theoretical normal distribution.")
            
        with tab_d2:
            fig_b = go.Figure()
            disp_tickers = [t for t in stock_tickers if t in active_returns.columns]
            if use_benchmark and BENCHMARK in active_returns.columns:
                disp_tickers.append(BENCHMARK)
                
            for t in disp_tickers:
                s = active_returns[t].dropna()
                if not s.empty:
                    fig_b.add_trace(go.Box(
                        y=s.values,
                        name=t,
                        boxpoints='outliers',
                        marker=dict(size=3, opacity=0.6),
                        line=dict(color='#315CEB' if t == selected_stock else ('#17243A' if t == BENCHMARK else '#69778C'), width=1.5)
                    ))
                    
            fig_b.add_hline(y=0, line_color="#69778C", line_width=1)
            fig_b.update_layout(create_clean_layout(
                title="Cross-Asset Return Dispersion & Extreme Outliers",
                xaxis_title="Asset",
                yaxis_title="Daily Return (%)",
                height=400,
                show_legend=False
            ))
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption("Points beyond the whiskers represent return sessions exceeding 1.5 × IQR from the first and third quartiles.")
            
        with tab_d3:
            n_tot = len(clean_ret)
            w1 = np.sum((clean_ret >= mu - sigma) & (clean_ret <= mu + sigma)) / n_tot * 100
            w2 = np.sum((clean_ret >= mu - 2*sigma) & (clean_ret <= mu + 2*sigma)) / n_tot * 100
            w3 = np.sum((clean_ret >= mu - 3*sigma) & (clean_ret <= mu + 3*sigma)) / n_tot * 100
            
            df_rule = pd.DataFrame({
                "Interval": ["Within ±1 Standard Deviation", "Within ±2 Standard Deviations", "Within ±3 Standard Deviations"],
                "Return Range (%)": [
                    f"[{mu - sigma:.2f}%, {mu + sigma:.2f}%]",
                    f"[{mu - 2*sigma:.2f}%, {mu + 2*sigma:.2f}%]",
                    f"[{mu - 3*sigma:.2f}%, {mu + 3*sigma:.2f}%]"
                ],
                "Observed In Data": [f"{w1:.2f}%", f"{w2:.2f}%", f"{w3:.2f}%"],
                "Normal Rule (Gaussian)": ["68.27%", "95.45%", "99.73%"],
                "Chebyshev Guarantee": ["≥ 0.0%", "≥ 75.0%", "≥ 88.9%"]
            })
            st.markdown("### Empirical Rule (68-95-99.7) vs. Chebyshev Bound Evaluation")
            st.dataframe(df_rule, hide_index=True, use_container_width=True)
            st.caption(
                "Chebyshev's inequality guarantees absolute mathematical minimum bounds for any arbitrary distribution, "
                "whereas observed daily stock returns closely track Gaussian benchmarks with slight tail deviations."
            )


# ==========================================
# 5. SAMPLING & CLT
# ==========================================
elif nav_selection == "Sampling & CLT":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Sampling & Central Limit Theorem</div>
        <div class="page-desc">Compare probability sampling strategies and verify sampling distribution convergence.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("Insufficient data available for sampling analysis.")
    else:
        population = stock_return_series.to_numpy()
        N = len(population)
        pop_mean = float(np.mean(population))
        pop_std = float(np.std(population, ddof=1))
        
        tab_s1, tab_s2 = st.tabs([
            "Probability Sampling Methods",
            "Central Limit Theorem Simulation"
        ])
        
        with tab_s1:
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                n_sample = st.slider("Sample Size (n)", 10, min(150, N // 2), 30, 5)
            with col_sc2:
                seed = st.number_input("Random Seed", 1, 9999, 42)
                
            # 1. Simple Random Sampling (SRS)
            srs = stock_return_series.sample(n=n_sample, random_state=seed)
            # 2. Systematic Sampling (1-in-k)
            k = max(1, N // n_sample)
            syst = stock_return_series.iloc[::k].head(n_sample)
            # 3. Stratified by Year
            df_yr = stock_return_series.rename("Return").to_frame()
            df_yr["Year"] = df_yr.index.year
            quota = max(1, int(np.ceil(n_sample / df_yr["Year"].nunique())))
            strat = (df_yr.groupby("Year", group_keys=False).apply(
                lambda g: g.sample(min(len(g), quota), random_state=seed)
            ))["Return"].head(n_sample)
            
            samp_df = pd.DataFrame([
                {"Sampling Method": "Full Population (Ground Truth)", "Sample Size (n)": N, "Mean Return (%)": f"{pop_mean:.4f}%", "Std Dev (%)": f"{pop_std:.4f}%", "Error from μ": "0.0000%"},
                {"Sampling Method": "Simple Random Sampling (SRS)", "Sample Size (n)": len(srs), "Mean Return (%)": f"{srs.mean():.4f}%", "Std Dev (%)": f"{srs.std(ddof=1):.4f}%", "Error from μ": f"{abs(srs.mean()-pop_mean):.4f}%"},
                {"Sampling Method": f"Systematic Sampling (k={k})", "Sample Size (n)": len(syst), "Mean Return (%)": f"{syst.mean():.4f}%", "Std Dev (%)": f"{syst.std(ddof=1):.4f}%", "Error from μ": f"{abs(syst.mean()-pop_mean):.4f}%"},
                {"Sampling Method": "Stratified Sampling (by Year)", "Sample Size (n)": len(strat), "Mean Return (%)": f"{strat.mean():.4f}%", "Std Dev (%)": f"{strat.std(ddof=1):.4f}%", "Error from μ": f"{abs(strat.mean()-pop_mean):.4f}%"}
            ])
            st.markdown("### Sampling Method Comparison against Ground Truth")
            st.dataframe(samp_df, hide_index=True, use_container_width=True)
            st.caption("Each sampling scheme yields an unbiased estimator of the true population mean return, with stratified sampling reducing variance across multi-year macroeconomic regimes.")
            
        with tab_s2:
            st.markdown("### Interactive Simulation of the Central Limit Theorem")
            st.caption("As sample size (n) increases, the distribution of sample averages approaches a normal distribution with standard error σ/√n regardless of population skewness.")
            
            col_cl1, col_cl2 = st.columns(2)
            with col_cl1:
                clt_n = st.select_slider("Sample Size (n per draw)", options=[5, 15, 30, 50, 100], value=30)
            with col_cl2:
                clt_runs = st.select_slider("Monte Carlo Simulation Draws (M)", options=[500, 1000, 2000], value=1000)
                
            np.random.seed(42)
            sample_means = np.array([
                np.mean(np.random.choice(population, size=clt_n, replace=True))
                for _ in range(clt_runs)
            ])
            
            mean_of_means = float(np.mean(sample_means))
            empirical_se = float(np.std(sample_means, ddof=1))
            theoretical_se = pop_std / np.sqrt(clt_n)
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Population Mean (μ)", f"{pop_mean:.4f}%")
            with r2:
                st.metric("Simulated Mean of Means", f"{mean_of_means:.4f}%", f"{mean_of_means - pop_mean:+.4f}%")
            with r3:
                st.metric("Theoretical Standard Error (σ/√n)", f"{theoretical_se:.4f}%", f"Observed SE: {empirical_se:.4f}%")
                
            fig_clt = go.Figure()
            fig_clt.add_trace(go.Histogram(
                x=sample_means,
                histnorm='probability density',
                nbinsx=40,
                name="Simulated Sample Means",
                marker_color='#315CEB',
                opacity=0.65,
                hovertemplate="Sample Mean: %{x:.4f}%<br>Density: %{y:.2f}<extra></extra>"
            ))
            
            x_vals = np.linspace(sample_means.min(), sample_means.max(), 150)
            clt_pdf = stats.norm.pdf(x_vals, loc=pop_mean, scale=theoretical_se)
            fig_clt.add_trace(go.Scatter(
                x=x_vals,
                y=clt_pdf,
                mode='lines',
                name=f"Theoretical Normal N(μ, σ/√n)",
                line=dict(color='#F59E0B', width=2.2),
                hovertemplate="Mean: %{x:.4f}%<br>Normal Curve: %{y:.2f}<extra></extra>"
            ))
            fig_clt.add_vline(x=pop_mean, line_dash="dash", line_color="#D94F59", annotation_text="μ (Pop Mean)", annotation_position="top right")
            
            fig_clt.update_layout(create_clean_layout(
                title=f"Sampling Distribution of Mean Daily Returns (n={clt_n}, M={clt_runs} draws)",
                xaxis_title="Sample Mean (%)",
                yaxis_title="Probability Density",
                height=390
            ))
            st.plotly_chart(fig_clt, use_container_width=True)
            st.caption(
                f"Plain English Takeaway: Even though individual daily returns for {selected_stock} feature extreme shocks and heavy tails, "
                f"the average of {clt_n} randomly chosen sessions behaves in near-perfect accordance with Gaussian distribution theory."
            )


# ==========================================
# 6. HYPOTHESIS TESTING
# ==========================================
elif nav_selection == "Hypothesis Testing":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Hypothesis Testing</div>
        <div class="page-desc">Test statistical hypotheses regarding expected returns, asset equality, and market directional symmetry.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("Insufficient data available for hypothesis testing.")
    else:
        tab_h1, tab_h2, tab_h3, tab_h4 = st.tabs([
            "One-Sample & Z Tests",
            "Confidence Intervals",
            "Two-Sample & Paired Tests",
            "Chi-Square Tests"
        ])
        
        n_obs = len(stock_return_series)
        x_bar = float(stock_return_series.mean())
        s_std = float(stock_return_series.std(ddof=1))
        se = s_std / np.sqrt(n_obs)
        
        with tab_h1:
            st.markdown("### One-Sample Test for Zero Expected Daily Return")
            st.caption("Null Hypothesis (H0): Population mean daily return μ = 0.0% vs. Alternative Hypothesis (H1): μ ≠ 0.0%")
            
            alpha = st.selectbox("Significance Level (α)", options=[0.05, 0.01, 0.10], index=0)
            
            t_stat, t_pval = stats.ttest_1samp(stock_return_series, popmean=0.0)
            z_stat = (x_bar - 0.0) / se
            z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            col_th1, col_th2 = st.columns(2)
            with col_th1:
                t_is_sig = t_pval < alpha
                t_concl = "Reject H0 (Statistically Significant)" if t_is_sig else "Fail to Reject H0 (Insignificant)"
                t_badge = "badge-pos" if t_is_sig else "badge-neutral"
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">One-Sample Student's t-Test</div>
                    <div style="font-size: 0.88rem; line-height: 1.85;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Null Hypothesis (H0):</span>
                            <strong>μ = 0.0%</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">t-Statistic:</span>
                            <strong>{t_stat:.4f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Degrees of Freedom:</span>
                            <strong>{n_obs - 1}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Two-Sided p-Value:</span>
                            <strong>{t_pval:.5f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #69778C;">Decision (α={alpha}):</span>
                            <span class="{t_badge}">{t_concl}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_th2:
                z_is_sig = z_pval < alpha
                z_concl = "Reject H0 (Statistically Significant)" if z_is_sig else "Fail to Reject H0"
                z_badge = "badge-pos" if z_is_sig else "badge-neutral"
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Large-Sample Z-Test Approximation</div>
                    <div style="font-size: 0.88rem; line-height: 1.85;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Null Hypothesis (H0):</span>
                            <strong>μ = 0.0%</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">z-Statistic:</span>
                            <strong>{z_stat:.4f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Sample Size (n):</span>
                            <strong>{n_obs:,}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Asymptotic p-Value:</span>
                            <strong>{z_pval:.5f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #69778C;">Decision (α={alpha}):</span>
                            <span class="{z_badge}">{z_concl}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.caption(
                "Methodological Note: With sample sizes exceeding 1,000 sessions, the t and z distributions converge identically. "
                "Failing to reject H0 does not prove that expected returns are exactly zero; it demonstrates that observed drift is within normal sampling error."
            )
            
        with tab_h2:
            st.markdown("### Two-Sided Confidence Intervals for Mean Daily Return")
            ci_rows = []
            for conf in [0.90, 0.95, 0.99]:
                a = 1 - conf
                t_crit = stats.t.ppf(1 - a / 2, df=n_obs - 1)
                moe = t_crit * se
                low_b = x_bar - moe
                up_b = x_bar + moe
                contains_zero = low_b <= 0 <= up_b
                ci_rows.append({
                    "Confidence Level": f"{int(conf*100)}%",
                    "Critical Value (t*)": round(t_crit, 4),
                    "Margin of Error": f"±{moe:.4f}%",
                    "Lower Bound": f"{low_b:+.4f}%",
                    "Upper Bound": f"{up_b:+.4f}%",
                    "Contains Zero (μ=0)?": "Yes (Consistent with 0)" if contains_zero else "No (Statistically Distinct)"
                })
            st.dataframe(pd.DataFrame(ci_rows), hide_index=True, use_container_width=True)
            
        with tab_h3:
            st.markdown("### Two-Sample & Paired Tests Between Assets")
            other_stocks = [t for t in stock_tickers if t != selected_stock]
            if other_stocks:
                comp_stock = st.selectbox("Select Asset to Compare:", other_stocks)
                s_comp = active_returns[comp_stock].dropna()
                
                # Independent Welch's
                welch = stats.ttest_ind(stock_return_series, s_comp, equal_var=False)
                # Paired
                paired = active_returns[[selected_stock, comp_stock]].dropna()
                rel = stats.ttest_rel(paired[selected_stock], paired[comp_stock])
                
                c_w1, c_w2 = st.columns(2)
                with c_w1:
                    w_sig = welch.pvalue < 0.05
                    st.markdown(f"""
                    <div class="content-card">
                        <div class="card-title">Welch's Two-Sample t-Test</div>
                        <div style="font-size: 0.88rem; line-height: 1.85;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">H0:</span>
                                <strong>μ({selected_stock}) = μ({comp_stock})</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">t-Statistic:</span>
                                <strong>{welch.statistic:.4f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">p-Value:</span>
                                <strong>{welch.pvalue:.5f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                                <span style="color: #69778C;">Conclusion (α=0.05):</span>
                                <span class="{'badge-pos' if w_sig else 'badge-neutral'}">{'Statistically Significant Difference' if w_sig else 'No Significant Difference'}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c_w2:
                    p_sig = rel.pvalue < 0.05
                    st.markdown(f"""
                    <div class="content-card">
                        <div class="card-title">Paired t-Test (Synchronous Trading Sessions)</div>
                        <div style="font-size: 0.88rem; line-height: 1.85;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">H0:</span>
                                <strong>Mean Daily Difference = 0</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">t-Statistic:</span>
                                <strong>{rel.statistic:.4f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                                <span style="color: #69778C;">p-Value:</span>
                                <strong>{rel.pvalue:.5f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                                <span style="color: #69778C;">Synchronous Days:</span>
                                <strong>{len(paired):,}</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with tab_h4:
            st.markdown("### Chi-Square Categorical Tests")
            
            # 1. Goodness of Fit (Direction)
            pos_days = int((stock_return_series > 0).sum())
            non_pos_days = int((stock_return_series <= 0).sum())
            chi2_gof, p_gof = stats.chisquare([pos_days, non_pos_days], [n_obs / 2, n_obs / 2])
            
            # 2. Weekday Independence
            df_w = stock_return_series.rename("Return").to_frame()
            df_w["Weekday"] = df_w.index.day_name()
            df_w["Direction"] = np.where(df_w["Return"] > 0, "Positive", "Non-positive")
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            ct = pd.crosstab(df_w["Weekday"], df_w["Direction"]).reindex([d for d in days_order if d in df_w["Weekday"].values])
            chi2_w, p_w, dof_w, _ = stats.chi2_contingency(ct)
            
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                gof_sig = p_gof < 0.05
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Directional Goodness-of-Fit Test</div>
                    <div style="font-size: 0.88rem; line-height: 1.85;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Null Hypothesis (H0):</span>
                            <strong>P(Up) = P(Down) = 0.50</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Positive Sessions:</span>
                            <strong>{pos_days} ({pos_days/n_obs*100:.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Chi² Statistic:</span>
                            <strong>{chi2_gof:.4f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">p-Value:</span>
                            <strong>{p_gof:.5f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #69778C;">Result (α=0.05):</span>
                            <span class="{'badge-pos' if gof_sig else 'badge-neutral'}">{'Statistically Significant Up Bias' if gof_sig else 'Consistent with 50/50 Chance'}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_ch2:
                w_is_sig = p_w < 0.05
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Weekday Independence (Calendar Anomaly)</div>
                    <div style="font-size: 0.88rem; line-height: 1.85;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Null Hypothesis (H0):</span>
                            <strong>Direction is independent of weekday</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Chi² Statistic (df={dof_w}):</span>
                            <strong>{chi2_w:.4f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">p-Value:</span>
                            <strong>{p_w:.5f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #69778C;">Result (α=0.05):</span>
                            <span class="{'badge-pos' if w_is_sig else 'badge-neutral'}">{'Weekday Anomaly Detected' if w_is_sig else 'No Weekday Effect'}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### Contingency Table: Weekday vs. Return Direction")
            st.dataframe(ct, use_container_width=True)


# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================
elif nav_selection == "Correlation Analysis":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Correlation Analysis</div>
        <div class="page-desc">Measure linear co-movement, rank associations, and systematic market risk (Beta).</div>
    </div>
    """, unsafe_allow_html=True)
    
    corr_cols = [c for c in stock_tickers if c in active_returns.columns]
    if use_benchmark and BENCHMARK in active_returns.columns:
        corr_cols.append(BENCHMARK)
    df_corr = active_returns[corr_cols].dropna()
    
    if df_corr.empty or len(corr_cols) < 2:
        st.warning("Insufficient multi-asset data available for correlation analysis.")
    else:
        tab_cr1, tab_cr2, tab_cr3 = st.tabs([
            "Correlation Matrices",
            "Stock vs. Benchmark Scatter (Beta)",
            "Pairwise Stock Scatter"
        ])
        
        with tab_cr1:
            m_type = st.radio("Metric", ["Pearson Correlation (Linear)", "Spearman Correlation (Rank)", "Covariance"], horizontal=True)
            
            if "Pearson" in m_type:
                mat = df_corr.corr(method="pearson")
                c_title = "Pearson Linear Correlation Matrix"
                zmin, zmax = -1.0, 1.0
                # Diverging scale centered at 0: Negative #D94F59, Zero #FFFFFF, Positive #315CEB
                cscale = [[0.0, "#D94F59"], [0.5, "#FFFFFF"], [1.0, "#315CEB"]]
            elif "Spearman" in m_type:
                mat = df_corr.corr(method="spearman")
                c_title = "Spearman Monotonic Rank Correlation Matrix"
                zmin, zmax = -1.0, 1.0
                cscale = [[0.0, "#D94F59"], [0.5, "#FFFFFF"], [1.0, "#315CEB"]]
            else:
                mat = df_corr.cov()
                c_title = "Empirical Covariance Matrix (%²)"
                zmin, zmax = None, None
                cscale = "Blues"
                
            fig_hm = go.Figure(data=go.Heatmap(
                z=mat.values,
                x=mat.columns,
                y=mat.index,
                colorscale=cscale,
                zmin=zmin,
                zmax=zmax,
                text=np.round(mat.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=11, family="Inter, sans-serif"),
                hovertemplate="<b>%{x} vs. %{y}</b><br>Value: %{z:.4f}<extra></extra>"
            ))
            fig_hm.update_layout(create_clean_layout(
                title=c_title,
                height=420
            ))
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("Diverging color gradient: blue indicates positive co-movement, red indicates inverse diversification, and white indicates near-zero correlation.")
            
        with tab_cr2:
            if BENCHMARK not in active_returns.columns:
                st.info("Benchmark data (^GSPC) not found.")
            else:
                paired = active_returns[[selected_stock, BENCHMARK]].dropna()
                x_val = paired[BENCHMARK]
                y_val = paired[selected_stock]
                
                slope, intercept, r_val, p_val, std_err = stats.linregress(x_val, y_val)
                
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("Beta (β - Systematic Risk)", f"{slope:.3f}", help="Sensitivity to broader S&P 500 market movements.")
                with col_b2:
                    st.metric("Alpha (α - Daily Intercept)", f"{intercept:+.4f}%", help="Unsystematic daily excess return.")
                with col_b3:
                    st.metric("R² (Variance Explained)", f"{r_val**2:.3f}", f"Correlation (r): {r_val:.3f}")
                    
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=x_val,
                    y=y_val,
                    mode='markers',
                    name="Daily Sessions",
                    marker=dict(color='#315CEB', size=5, opacity=0.45),
                    hovertemplate="S&P 500: %{x:.2f}%<br>" + f"{selected_stock}: " + "%{y:.2f}%<extra></extra>"
                ))
                
                x_line = np.linspace(x_val.min(), x_val.max(), 100)
                fig_sc.add_trace(go.Scatter(
                    x=x_line,
                    y=intercept + slope * x_line,
                    mode='lines',
                    name=f"OLS Line (β={slope:.2f}, R²={r_val**2:.2f})",
                    line=dict(color='#F59E0B', width=2.2)
                ))
                
                fig_sc.add_hline(y=0, line_color="#E6EAF1", line_width=1)
                fig_sc.add_vline(x=0, line_color="#E6EAF1", line_width=1)
                
                fig_sc.update_layout(create_clean_layout(
                    title=f"{selected_stock} vs. S&P 500 Benchmark Return Scatter & OLS Fit",
                    xaxis_title="S&P 500 Daily Return (%)",
                    yaxis_title=f"{selected_stock} Daily Return (%)",
                    height=410
                ))
                st.plotly_chart(fig_sc, use_container_width=True)
                
        with tab_cr3:
            other_stocks = [t for t in stock_tickers if t != selected_stock]
            if other_stocks:
                pair_target = st.selectbox("Select Asset to Compare Pairwise:", other_stocks)
                pair_data = active_returns[[selected_stock, pair_target]].dropna()
                pr_r, pr_p = stats.pearsonr(pair_data[selected_stock], pair_data[pair_target])
                
                fig_p = go.Figure(go.Scatter(
                    x=pair_data[pair_target],
                    y=pair_data[selected_stock],
                    mode='markers',
                    marker=dict(color='#315CEB', size=5, opacity=0.45),
                    hovertemplate=f"{pair_target}: " + "%{x:.2f}%<br>" + f"{selected_stock}: " + "%{y:.2f}%<extra></extra>"
                ))
                fig_p.add_hline(y=0, line_color="#E6EAF1", line_width=1)
                fig_p.add_vline(x=0, line_color="#E6EAF1", line_width=1)
                
                fig_p.update_layout(create_clean_layout(
                    title=f"Pairwise Scatter: {selected_stock} vs. {pair_target} (r = {pr_r:.3f}, p = {pr_p:.1e})",
                    xaxis_title=f"{pair_target} Daily Return (%)",
                    yaxis_title=f"{selected_stock} Daily Return (%)",
                    height=380,
                    show_legend=False
                ))
                st.plotly_chart(fig_p, use_container_width=True)


# ==========================================
# 8. MODEL PERFORMANCE
# ==========================================
elif nav_selection == "Model Performance":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model Performance</div>
        <div class="page-desc">Review out-of-sample historical evaluation benchmarks across trained predictive algorithms.</div>
    </div>
    """, unsafe_allow_html=True)
    
    df_reg = data_dict.get("regression_comparison")
    df_clf = data_dict.get("classification_comparison")
    df_cm = data_dict.get("confusion_matrix")
    
    tab_mp1, tab_mp2, tab_mp3 = st.tabs([
        "Regression Models",
        "Classification Models",
        "Holdout Confusion Matrix"
    ])
    
    with tab_mp1:
        if df_reg is not None:
            st.markdown("### Next-Day Return Regression Leaderboard")
            st.caption("Target: Next-day percentage return. Evaluated on a chronological 20% holdout test set (unseen future sessions):")
            st.dataframe(df_reg.round(4), hide_index=True, use_container_width=True)
            
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(
                x=df_reg["Model"],
                y=df_reg["RMSE"],
                name="RMSE (% - Root Mean Squared Error)",
                marker_color='#315CEB'
            ))
            fig_reg.add_trace(go.Bar(
                x=df_reg["Model"],
                y=df_reg["MAE"],
                name="MAE (% - Mean Absolute Error)",
                marker_color='#69778C'
            ))
            fig_reg.update_layout(create_clean_layout(
                title="Regression Model Error Comparison (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="Error (%)",
                height=350
            ))
            st.plotly_chart(fig_reg, use_container_width=True)
            
            with st.expander("Academic Note: Why are regression R² scores near zero?"):
                st.caption(
                    "In line with the Efficient Market Hypothesis (EMH), short-term daily financial returns exhibit very low signal-to-noise ratios. "
                    "While simple historical lags capture broad drift, predicting exact daily returns with high R² is inherently constrained by market efficiency."
                )
                
    with tab_mp2:
        if df_clf is not None:
            st.markdown("### Market Direction Classification Leaderboard")
            st.caption("Target: Next-day return direction (1 = Positive/Up, 0 = Non-positive/Down). Evaluated on 20% chronological holdout:")
            st.dataframe(df_clf.round(4), hide_index=True, use_container_width=True)
            
            fig_clf = go.Figure()
            fig_clf.add_trace(go.Bar(
                x=df_clf["Model"],
                y=df_clf["Accuracy"],
                name="Accuracy",
                marker_color='#315CEB'
            ))
            fig_clf.add_trace(go.Bar(
                x=df_clf["Model"],
                y=df_clf["F1-Score"],
                name="F1-Score",
                marker_color='#16856B'
            ))
            fig_clf.update_layout(create_clean_layout(
                title="Classification Performance Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score (0 to 1.0)",
                height=350
            ))
            st.plotly_chart(fig_clf, use_container_width=True)
            
    with tab_mp3:
        if df_cm is not None:
            col_m1, col_m2 = st.columns([1, 1.2])
            with col_m1:
                fig_cmat = go.Figure(data=go.Heatmap(
                    z=df_cm.values,
                    x=["Predicted Down (0)", "Predicted Up (1)"],
                    y=["Actual Down (0)", "Actual Up (1)"],
                    colorscale="Blues",
                    text=df_cm.values,
                    texttemplate="%{text}",
                    textfont=dict(size=14, family="Plus Jakarta Sans, sans-serif")
                ))
                fig_cmat.update_layout(create_clean_layout(
                    title="Test Set Confusion Matrix (Logistic Regression)",
                    height=340,
                    show_legend=False
                ))
                st.plotly_chart(fig_cmat, use_container_width=True)
                
            with col_m2:
                tn = df_cm.iloc[0, 0]
                fp = df_cm.iloc[0, 1]
                fn = df_cm.iloc[1, 0]
                tp = df_cm.iloc[1, 1]
                tot = tn + fp + fn + tp
                
                acc = (tn + tp) / tot * 100
                prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Diagnostic Breakdown</div>
                    <div style="font-size: 0.88rem; line-height: 1.85;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">Chronological Holdout Sessions:</span>
                            <strong>{tot} sessions</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">True Negatives (Correct Down):</span>
                            <strong>{tn}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">False Positives (Type I Error):</span>
                            <strong>{fp}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">False Negatives (Type II Error):</span>
                            <strong>{fn}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F4F9; padding: 4px 0;">
                            <span style="color: #69778C;">True Positives (Correct Up):</span>
                            <strong>{tp}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #69778C;">Overall Test Accuracy:</span>
                            <strong style="color: #315CEB;">{acc:.2f}%</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("""
    <div class="disclaimer-card" style="margin-top: 16px;">
        <strong>Historical Evaluation Disclaimer:</strong> These metrics reflect backtested evaluation on historical holdout partitions. 
        Past predictive performance does not guarantee future financial accuracy.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 9. PREDICTIONS
# ==========================================
elif nav_selection == "Predictions":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predictions</div>
        <div class="page-desc">Generate statistical next-day return estimates and directional probabilities via trained Scikit-Learn pipelines.</div>
    </div>
    """, unsafe_allow_html=True)
    
    reg_model = model_dict.get("regression")
    clf_model = model_dict.get("classification")
    
    if reg_model is None or clf_model is None:
        st.error("Saved model pipelines (.pkl) could not be loaded from models/.")
    else:
        EXPECTED_FEATURES = [
            'Return_Lag1', 'Return_Lag2', 'Return_Lag5',
            'Volatility_5D', 'Volatility_20D',
            'Moving_Average_5D', 'Moving_Average_20D',
            'Benchmark_Return'
        ]
        
        st.markdown("""
        <div class="content-card" style="padding: 14px 18px; margin-bottom: 16px;">
            <div style="font-size: 0.88rem; color: #475569; line-height: 1.55;">
                <strong>Pipeline Configuration:</strong> The deployed models (<strong>Lasso Regression</strong> and <strong>Logistic Regression</strong>) 
                were trained using <strong>Apple (AAPL)</strong> historical features and S&P 500 benchmark data across 8 technical dimensions: 
                <code style="color: #315CEB; font-size: 0.82rem;">Return_Lag1, Return_Lag2, Return_Lag5, Volatility_5D, Volatility_20D, Moving_Average_5D, Moving_Average_20D, Benchmark_Return</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        p_mode = st.radio("Inference Input Source:", ["Latest Historical Features (Real Observed Data)", "Custom Scenario Simulator"], horizontal=True)
        
        feat_df = None
        ref_date = "Custom"
        
        if "Latest" in p_mode:
            if df_model_data is not None and not df_model_data.empty:
                feat_df = df_model_data[EXPECTED_FEATURES].iloc[[-1]].copy()
                ref_date = df_model_data.index[-1].strftime('%b %d, %Y') if isinstance(df_model_data.index[-1], pd.Timestamp) else str(df_model_data.index[-1])
            else:
                s_ret = df_returns[selected_stock].dropna()
                bm_ret = df_returns[BENCHMARK].dropna() if BENCHMARK in df_returns.columns else pd.Series(0, index=s_ret.index)
                temp = pd.DataFrame(index=s_ret.index)
                temp["Return"] = s_ret
                temp["Return_Lag1"] = temp["Return"].shift(1)
                temp["Return_Lag2"] = temp["Return"].shift(2)
                temp["Return_Lag5"] = temp["Return"].shift(5)
                temp["Volatility_5D"] = temp["Return"].rolling(5).std()
                temp["Volatility_20D"] = temp["Return"].rolling(20).std()
                temp["Moving_Average_5D"] = temp["Return"].rolling(5).mean()
                temp["Moving_Average_20D"] = temp["Return"].rolling(20).mean()
                temp["Benchmark_Return"] = bm_ret
                temp = temp.dropna()
                if not temp.empty:
                    feat_df = temp[EXPECTED_FEATURES].iloc[[-1]].copy()
                    ref_date = temp.index[-1].strftime('%b %d, %Y')
                    
            if feat_df is not None:
                st.markdown(f"#### Active Feature Vector as of {ref_date}")
                st.dataframe(feat_df.round(4), hide_index=True, use_container_width=True)
                
        else:
            st.markdown("#### Adjust Market Parameters for Scenario Inference")
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            with c_p1:
                f_l1 = st.slider("Return Lag 1 (%)", -8.0, 8.0, 0.5, 0.1)
                f_l2 = st.slider("Return Lag 2 (%)", -8.0, 8.0, -0.2, 0.1)
            with c_p2:
                f_l5 = st.slider("Return Lag 5 (%)", -8.0, 8.0, 0.8, 0.1)
                f_bm = st.slider("S&P 500 Return (%)", -5.0, 5.0, 0.3, 0.1)
            with c_p3:
                f_v5 = st.slider("5-Day Volatility (%)", 0.2, 6.0, 1.4, 0.1)
                f_v20 = st.slider("20-Day Volatility (%)", 0.2, 6.0, 1.7, 0.1)
            with c_p4:
                f_m5 = st.slider("5-Day MA Return (%)", -4.0, 4.0, 0.2, 0.1)
                f_m20 = st.slider("20-Day MA Return (%)", -4.0, 4.0, 0.1, 0.1)
                
            feat_df = pd.DataFrame([{
                'Return_Lag1': f_l1,
                'Return_Lag2': f_l2,
                'Return_Lag5': f_l5,
                'Volatility_5D': f_v5,
                'Volatility_20D': f_v20,
                'Moving_Average_5D': f_m5,
                'Moving_Average_20D': f_m20,
                'Benchmark_Return': f_bm
            }])[EXPECTED_FEATURES]
            
        if feat_df is not None:
            try:
                pred_ret = float(reg_model.predict(feat_df)[0])
                pred_dir = int(clf_model.predict(feat_df)[0])
                pred_probs = clf_model.predict_proba(feat_df)[0] if hasattr(clf_model, "predict_proba") else None
                
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.markdown("### Model Prediction Outputs")
                
                col_o1, col_o2, col_o3 = st.columns(3)
                with col_o1:
                    r_sign = "+" if pred_ret >= 0 else ""
                    b_color = "badge-pos" if pred_ret >= 0 else "badge-neg"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Next-Day Return</div>
                        <div class="metric-value" style="color: {'#16856B' if pred_ret >= 0 else '#D94F59'};">{r_sign}{pred_ret:.4f}%</div>
                        <div class="metric-sub"><span class="{b_color}">Lasso Regression Model</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_o2:
                    dir_text = "Positive (Up)" if pred_dir == 1 else "Non-positive (Down/Flat)"
                    dir_badge = "badge-pos" if pred_dir == 1 else "badge-neg"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Market Direction</div>
                        <div class="metric-value" style="font-size: 1.35rem;">{dir_text}</div>
                        <div class="metric-sub"><span class="{dir_badge}">Logistic Regression Model</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_o3:
                    prob_str = f"{pred_probs[1]*100:.1f}% Up / {pred_probs[0]*100:.1f}% Down" if pred_probs is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Directional Class Probabilities</div>
                        <div class="metric-value" style="font-size: 1.35rem;">{prob_str}</div>
                        <div class="metric-sub"><span class="badge-neutral">Probability Estimate</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Inference error: {e}")
                
        # Academic Disclaimer
        st.markdown("""
        <div class="disclaimer-card">
            <strong>⚠️ Academic Disclaimer:</strong> These predictions are generated by statistical machine learning models 
            trained on historical equity data solely for university research and quantitative demonstration. 
            Financial markets are non-stationary and subject to unforeseen macroeconomic shocks. 
            <strong>Under no circumstances should these estimates be construed as investment recommendations, financial advice, or trading signals.</strong>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("<hr style='margin: 32px 0 16px 0; border: none; border-top: 1px solid #E6EAF1;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #69778C; font-size: 0.8rem; padding-bottom: 12px;">
    <strong>Stock Market Return Statistical Analyzer</strong> • Academic Analytics Dashboard
</div>
""", unsafe_allow_html=True)

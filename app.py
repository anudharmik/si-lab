"""
Stock Market Return Statistical Analyzer
A modern, light-mode financial analytics dashboard designed for academic research
and quantitative market data exploration.
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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODERN LIGHT-MODE STYLING
# ==========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        background-color: #F7F8FA !important;
        color: #17243A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #17243A !important;
        letter-spacing: -0.015em;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Project Brand Badge */
    .brand-container {
        padding-bottom: 14px;
        margin-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
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
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 2px;
        font-weight: 500;
    }
    
    /* Main App Header */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .app-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #17243A;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin-top: 5px;
        margin-bottom: 0;
    }
    
    /* Section Headings */
    .page-header {
        margin-bottom: 20px;
    }
    .page-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 3px;
        letter-spacing: -0.02em;
    }
    .page-desc {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #17243A;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 8px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-weight: 600;
    }
    .badge-pos {
        color: #059669;
        background-color: #ECFDF5;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.78rem;
    }
    .badge-neg {
        color: #DC2626;
        background-color: #FEF2F2;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.78rem;
    }
    .badge-neutral {
        color: #475569;
        background-color: #F1F5F9;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.78rem;
    }
    
    /* Card Container */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 12px;
    }
    
    /* Observation Box */
    .observation-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3563E9;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .observation-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: #17243A;
        margin-bottom: 4px;
    }
    .observation-text {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
        margin: 0;
    }
    
    /* Academic Disclaimer */
    .disclaimer-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 24px;
        color: #475569;
        font-size: 0.84rem;
        line-height: 1.5;
    }
    
    /* Subtle Tab & Control Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 16px;
        color: #64748B;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #3563E9 !important;
        border-bottom-color: #3563E9 !important;
    }
    
    /* Clean Streamlit Dataframe Overrides */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PLOTLY MODERN LIGHT THEME HELPER
# ==========================================
def create_clean_layout(title="", xaxis_title="", yaxis_title="", height=380, hovermode="x unified"):
    """Creates a consistent, publication-ready light-mode layout for Plotly figures."""
    return go.Layout(
        title=dict(
            text=title,
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#17243A", weight=700),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#17243A", size=11),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11, color="#64748B")),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=10),
            showline=True,
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color="#64748B")),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=10),
            showline=True,
            linecolor="#E2E8F0"
        ),
        margin=dict(l=45, r=20, t=42, b=38),
        height=height,
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#64748B"),
            bgcolor="rgba(255,255,255,0.8)"
        )
    )


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
            df.set_index(date_col, inplace=True)
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
            df.set_index(date_col, inplace=True)
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
                df.set_index("Date", inplace=True)
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
# SIDEBAR CONTROLS & CLEAN NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-title">📈 Stock Return Analyzer</div>
        <div class="brand-subtitle">Statistical & Predictive Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Primary Equity Selection
    selected_stock = st.selectbox(
        "Target Stock",
        options=stock_tickers,
        index=stock_tickers.index(DEFAULT_TICKER) if DEFAULT_TICKER in stock_tickers else 0,
        help="Select the equity asset to analyze."
    )
    
    # 2. Benchmark Inclusion
    has_benchmark = BENCHMARK in df_returns.columns
    use_benchmark = st.checkbox(
        "Compare with S&P 500 (^GSPC)",
        value=has_benchmark,
        disabled=not has_benchmark,
        help="Include S&P 500 benchmark in charts and comparisons."
    )
    
    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
    
    # 3. Date Filter
    min_date = df_returns.index.min().to_pydatetime()
    max_date = df_returns.index.max().to_pydatetime()
    
    date_preset = st.radio(
        "Date Window",
        ["Full 5 Years", "Past 3 Years", "Past 1 Year", "Custom Range"],
        index=0
    )
    
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
    
    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
    
    # 4. Navigation Menu
    nav_selection = st.radio(
        "Section",
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

# Slice active data according to selected date filter
active_returns = df_returns.loc[(df_returns.index >= ts_start) & (df_returns.index <= ts_end)].copy()
active_prices = df_prices.loc[(df_prices.index >= ts_start) & (df_prices.index <= ts_end)].copy()

stock_return_series = active_returns[selected_stock].dropna()
stock_price_series = active_prices[selected_stock].dropna() if selected_stock in active_prices.columns else pd.Series()
benchmark_return_series = active_returns[BENCHMARK].dropna() if (use_benchmark and BENCHMARK in active_returns.columns) else None

latest_avail_date_str = df_prices.index.max().strftime("%B %d, %Y")


# ==========================================
# 1. OVERVIEW PAGE (HERO / LANDING)
# ==========================================
if nav_selection == "Overview":
    # Header area
    st.markdown(f"""
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div class="app-title">Stock Market Return Statistical Analyzer</div>
                <div class="app-subtitle">Explore stock performance, daily returns, statistical patterns, and prediction models.</div>
            </div>
            <div style="text-align: right;">
                <span class="badge-neutral" style="font-size: 0.8rem; font-weight: 500;">Data current as of {latest_avail_date_str}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty or stock_price_series.empty:
        st.warning("No price data available for the selected dates.")
    else:
        # Key metrics calculations
        p_start = float(stock_price_series.iloc[0])
        p_end = float(stock_price_series.iloc[-1])
        cumulative_return = ((p_end - p_start) / p_start) * 100
        
        avg_daily_return = float(stock_return_series.mean())
        daily_volatility = float(stock_return_series.std(ddof=1))
        annualized_return = avg_daily_return * 252
        annualized_volatility = daily_volatility * np.sqrt(252)
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0
        
        # Drawdown
        rolling_max = stock_price_series.cummax()
        drawdowns = (stock_price_series - rolling_max) / rolling_max * 100
        max_drawdown = float(drawdowns.min())
        
        # 4 Key Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Closing Price</div>
                <div class="metric-value">${p_end:,.2f}</div>
                <div class="metric-sub"><span class="badge-neutral">{selected_stock} • Base: ${p_start:,.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            cum_sign = "+" if cumulative_return >= 0 else ""
            badge_type = "badge-pos" if cumulative_return >= 0 else "badge-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cumulative Return</div>
                <div class="metric-value">{cum_sign}{cumulative_return:.2f}%</div>
                <div class="metric-sub"><span class="{badge_type}">{cum_sign}{cumulative_return:.1f}% over {len(stock_return_series):,} days</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            avg_sign = "+" if avg_daily_return >= 0 else ""
            badge_avg = "badge-pos" if avg_daily_return >= 0 else "badge-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Daily Return</div>
                <div class="metric-value">{avg_sign}{avg_daily_return:.3f}%</div>
                <div class="metric-sub"><span class="{badge_avg}">Annualized: {avg_sign}{annualized_return:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Volatility (s)</div>
                <div class="metric-value">{daily_volatility:.2f}%</div>
                <div class="metric-sub"><span class="badge-neutral">Annualized: {annualized_volatility:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        
        # MAIN CHART: Rebased Performance Comparison (Base = 100)
        st.markdown("### Historical Performance vs. S&P 500 (Base = 100)")
        
        norm_stock = (stock_price_series / p_start) * 100
        
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=norm_stock.index,
            y=norm_stock.values,
            mode='lines',
            name=f"{selected_stock} ({cum_sign}{cumulative_return:.1f}%)",
            line=dict(color='#3563E9', width=2.4),
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Normalized:</b> %{y:.1f}<extra></extra>"
        ))
        
        bm_cum = 0.0
        if benchmark_return_series is not None and BENCHMARK in active_prices.columns:
            bm_prices = active_prices[BENCHMARK].dropna()
            if not bm_prices.empty:
                bm_start = float(bm_prices.iloc[0])
                bm_end = float(bm_prices.iloc[-1])
                bm_cum = ((bm_end - bm_start) / bm_start) * 100
                norm_bm = (bm_prices / bm_start) * 100
                bm_sign = "+" if bm_cum >= 0 else ""
                
                fig_main.add_trace(go.Scatter(
                    x=norm_bm.index,
                    y=norm_bm.values,
                    mode='lines',
                    name=f"S&P 500 (^GSPC) ({bm_sign}{bm_cum:.1f}%)",
                    line=dict(color='#94A3B8', width=1.8, dash='dash'),
                    hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>S&P 500:</b> %{y:.1f}<extra></extra>"
                ))
                
        # Reference line at 100
        fig_main.add_hline(y=100, line_dash="dot", line_color="#CBD5E1", line_width=1)
        
        fig_main.update_layout(create_clean_layout(
            title="",
            xaxis_title="",
            yaxis_title="Normalized Value (Start = 100)",
            height=400
        ))
        st.plotly_chart(fig_main, use_container_width=True)
        
        # AUTOMATED FACTUAL OBSERVATION BOX
        stock_win_pct = float((stock_return_series > 0).mean() * 100)
        perf_comparison = "outperformed" if cumulative_return > bm_cum else "underperformed"
        
        observation_msg = (
            f"Over the {len(stock_return_series):,} trading sessions between {stock_price_series.index.min().strftime('%b %d, %Y')} "
            f"and {stock_price_series.index.max().strftime('%b %d, %Y')}, {selected_stock} generated a cumulative return of "
            f"{cum_sign}{cumulative_return:.1f}%, which {perf_comparison} the S&P 500 benchmark ({'+' if bm_cum >= 0 else ''}{bm_cum:.1f}%). "
            f"The asset exhibited an average daily return of {avg_sign}{avg_daily_return:.3f}% alongside a daily volatility of {daily_volatility:.2f}% "
            f"(annualized to {annualized_volatility:.1f}%), yielding a Sharpe ratio of {sharpe_ratio:.2f}. "
            f"Overall, {stock_win_pct:.1f}% of observed trading sessions closed with positive returns, with a peak-to-trough maximum drawdown of {max_drawdown:.1f}%."
        )
        
        st.markdown(f"""
        <div class="observation-card">
            <div class="observation-title">📌 Empirical Summary Observation</div>
            <div class="observation-text">{observation_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # ADDITIONAL OVERVIEW SECTION: Multi-Stock Comparison Table & Risk Summary
        col_tbl, col_risk = st.columns([1.6, 1])
        
        with col_tbl:
            st.markdown("### Available Stock Comparison")
            rows = []
            for t in stock_tickers:
                if t in active_returns.columns and t in active_prices.columns:
                    ret_s = active_returns[t].dropna()
                    prc_s = active_prices[t].dropna()
                    if not ret_s.empty and not prc_s.empty:
                        c_ret = ((prc_s.iloc[-1] - prc_s.iloc[0]) / prc_s.iloc[0]) * 100
                        d_vol = ret_s.std(ddof=1)
                        ann_vol = d_vol * np.sqrt(252)
                        rows.append({
                            "Stock": t,
                            "Latest Price": f"${prc_s.iloc[-1]:,.2f}",
                            "Cumulative Return": f"{'+' if c_ret >= 0 else ''}{c_ret:.1f}%",
                            "Avg Daily Return": f"{ret_s.mean():+.3f}%",
                            "Daily Volatility": f"{d_vol:.2f}%",
                            "Sharpe Ratio": f"{(ret_s.mean()*252)/ann_vol:.2f}" if ann_vol > 0 else "N/A"
                        })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            
        with col_risk:
            st.markdown("### Risk & Return Profile")
            st.markdown(f"""
            <div class="content-card" style="padding: 16px;">
                <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 8px;"><strong>Risk-Adjusted Metrics ({selected_stock})</strong></div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.88rem;">
                    <span>Annualized Return:</span>
                    <strong>{annualized_return:+.1f}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.88rem;">
                    <span>Annualized Volatility:</span>
                    <strong>{annualized_volatility:.1f}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.88rem;">
                    <span>Sharpe Ratio (Rf=0%):</span>
                    <strong>{sharpe_ratio:.2f}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.88rem;">
                    <span>Maximum Drawdown:</span>
                    <strong style="color: #DC2626;">{max_drawdown:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# 2. HISTORICAL TRENDS
# ==========================================
elif nav_selection == "Historical Trends":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Historical Trends</div>
        <div class="page-desc">Track historical prices, moving averages, and cumulative portfolio growth.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_price_series.empty or stock_return_series.empty:
        st.warning("Insufficient data available.")
    else:
        tab_t1, tab_t2, tab_t3 = st.tabs(["Price Action & Moving Averages", "Growth of $10,000", "Daily Returns & Drawdown"])
        
        with tab_t1:
            ma20 = stock_price_series.rolling(20).mean()
            ma50 = stock_price_series.rolling(50).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_price_series.index, y=stock_price_series.values, mode='lines', name=f"{selected_stock} Close", line=dict(color='#3563E9', width=2.2)))
            fig.add_trace(go.Scatter(x=ma20.index, y=ma20.values, mode='lines', name="20-Day MA", line=dict(color='#F59E0B', width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode='lines', name="50-Day MA", line=dict(color='#10B981', width=1.5, dash='dot')))
            
            fig.update_layout(create_clean_layout(
                title=f"{selected_stock} Adjusted Closing Price with 20-Day and 50-Day Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($ USD)",
                height=420
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("About Moving Averages"):
                st.caption("A 20-day moving average tracks short-term momentum, while a 50-day moving average highlights medium-term market trends by smoothing daily fluctuations.")
                
        with tab_t2:
            growth_stock = (1 + stock_return_series / 100).cumprod() * 10000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth_stock.index, y=growth_stock.values, mode='lines', name=f"{selected_stock} Portfolio", line=dict(color='#3563E9', width=2.4)))
            
            if benchmark_return_series is not None and not benchmark_return_series.empty:
                growth_bm = (1 + benchmark_return_series / 100).cumprod() * 10000
                fig.add_trace(go.Scatter(x=growth_bm.index, y=growth_bm.values, mode='lines', name="S&P 500 (^GSPC)", line=dict(color='#94A3B8', width=1.8, dash='dash')))
                
            fig.add_hline(y=10000, line_dash="dot", line_color="#CBD5E1", annotation_text="Initial $10,000", annotation_position="bottom right")
            
            fig.update_layout(create_clean_layout(
                title=f"Cumulative Growth of an Initial $10,000 Investment",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($ USD)",
                height=420
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            final_val = float(growth_stock.iloc[-1])
            st.caption(f"A $10,000 investment in {selected_stock} on {stock_return_series.index.min().strftime('%b %d, %Y')} would be worth **${final_val:,.2f}** on {stock_return_series.index.max().strftime('%b %d, %Y')}.")
            
        with tab_t3:
            col_bar, col_dd = st.columns(2)
            
            with col_bar:
                colors = ['#10B981' if v >= 0 else '#EF4444' for v in stock_return_series.values]
                fig_ret = go.Figure(go.Bar(
                    x=stock_return_series.index,
                    y=stock_return_series.values,
                    marker_color=colors,
                    name="Daily Return"
                ))
                fig_ret.update_layout(create_clean_layout(
                    title=f"{selected_stock} Daily Percentage Return Fluctuations",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=360
                ))
                st.plotly_chart(fig_ret, use_container_width=True)
                
            with col_dd:
                rolling_max = stock_price_series.cummax()
                drawdown = (stock_price_series - rolling_max) / rolling_max * 100
                fig_dd = go.Figure(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    line=dict(color='#EF4444', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.1)',
                    name="Drawdown"
                ))
                fig_dd.update_layout(create_clean_layout(
                    title=f"{selected_stock} Drawdown from Peak",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=360
                ))
                st.plotly_chart(fig_dd, use_container_width=True)


# ==========================================
# 3. STATISTICS
# ==========================================
elif nav_selection == "Statistics":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Descriptive Statistics</div>
        <div class="page-desc">Understand the typical return and how much returns vary.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("No data points available.")
    else:
        stats_series = calculate_descriptive_stats(stock_return_series)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Central Tendency</div>
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    <strong>Mean:</strong> {stats_series['Mean (%)']:+.4f}%<br>
                    <strong>Median:</strong> {stats_series['Median (%)']:+.4f}%<br>
                    <strong>Mode:</strong> {stats_series['Mode (%)']:+.4f}%<br>
                    <strong>Observations:</strong> {int(stats_series['Count']):,}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Spread & Dispersion</div>
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    <strong>Standard Deviation:</strong> {stats_series['Std Deviation (%)']:.4f}%<br>
                    <strong>Variance:</strong> {stats_series['Variance (%²)']:.4f}%²<br>
                    <strong>Interquartile Range:</strong> {stats_series['IQR (%)']:.4f}%<br>
                    <strong>Range:</strong> [{stats_series['Minimum (%)']:.2f}%, {stats_series['Maximum (%)']:.2f}%]
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s3:
            skew_val = stats_series['Skewness']
            kurt_val = stats_series['Kurtosis']
            st.markdown(f"""
            <div class="content-card">
                <div class="card-title">Distribution Shape</div>
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    <strong>Skewness:</strong> {skew_val:+.4f}<br>
                    <strong>Excess Kurtosis:</strong> {kurt_val:.4f}<br>
                    <strong>Q1 (25th):</strong> {stats_series['Q1 (25th %)']:+.4f}%<br>
                    <strong>Q3 (75th):</strong> {stats_series['Q3 (75th %)']:+.4f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Cross-asset comparison table
        st.markdown("### Cross-Stock Statistical Comparison")
        all_stats = {}
        for t in stock_tickers:
            if t in active_returns.columns:
                s = active_returns[t].dropna()
                if not s.empty:
                    all_stats[t] = calculate_descriptive_stats(s)
        if use_benchmark and BENCHMARK in active_returns.columns:
            s_bm = active_returns[BENCHMARK].dropna()
            if not s_bm.empty:
                all_stats[BENCHMARK] = calculate_descriptive_stats(s_bm)
                
        df_all_stats = pd.DataFrame(all_stats).T
        st.dataframe(df_all_stats.round(4), use_container_width=True)
        
        with st.expander("Statistical Concept Reference"):
            st.markdown("""
            - **Skewness:** Measures symmetry. Positive values indicate a longer right tail (infrequent large gains), while negative values indicate a longer left tail.
            - **Excess Kurtosis:** Measures tail weight relative to a normal distribution (0). Positive kurtosis (leptokurtosis) indicates 'fat tails' with higher probabilities of extreme market moves.
            - **IQR (Interquartile Range):** The difference between the 75th and 25th percentiles, covering the middle 50% of return outcomes.
            """)


# ==========================================
# 4. RETURN DISTRIBUTION
# ==========================================
elif nav_selection == "Return Distribution":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Return Distribution</div>
        <div class="page-desc">See how returns are spread and identify unusually large movements.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("No data points available.")
    else:
        clean_ret = stock_return_series.to_numpy()
        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        
        tab_dist1, tab_dist2, tab_dist3 = st.tabs(["Histogram & Normal Fit", "Box Plot & Outliers", "Empirical Rule vs. Chebyshev"])
        
        with tab_dist1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=clean_ret,
                histnorm='probability density',
                nbinsx=50,
                name="Observed Returns",
                marker_color='#3563E9',
                opacity=0.65
            ))
            
            x_range = np.linspace(clean_ret.min() - 0.5, clean_ret.max() + 0.5, 200)
            pdf_vals = stats.norm.pdf(x_range, loc=mu, scale=sigma)
            fig_hist.add_trace(go.Scatter(
                x=x_range,
                y=pdf_vals,
                mode='lines',
                name=f"Normal Fit (μ={mu:.2f}%, s={sigma:.2f}%)",
                line=dict(color='#F59E0B', width=2.2)
            ))
            
            fig_hist.add_vline(x=mu, line_dash="dash", line_color="#10B981", annotation_text=f"Mean ({mu:.2f}%)")
            fig_hist.update_layout(create_clean_layout(
                title=f"{selected_stock} Daily Return Distribution vs. Normal Distribution Curve",
                xaxis_title="Daily Return (%)",
                yaxis_title="Density",
                height=420
            ))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab_dist2:
            fig_box = go.Figure()
            for t in stock_tickers:
                if t in active_returns.columns:
                    s = active_returns[t].dropna()
                    if not s.empty:
                        fig_box.add_trace(go.Box(
                            y=s.values,
                            name=t,
                            boxpoints='outliers',
                            marker=dict(size=3, opacity=0.6),
                            line_width=1.5
                        ))
            fig_box.update_layout(create_clean_layout(
                title="Return Dispersion & Extreme Movement Outliers by Stock",
                xaxis_title="Ticker",
                yaxis_title="Daily Return (%)",
                height=400
            ))
            st.plotly_chart(fig_box, use_container_width=True)
            
        with tab_dist3:
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
                "Normal Rule Benchmark": ["68.27%", "95.45%", "99.73%"],
                "Chebyshev Guarantee": ["≥ 0.0%", "≥ 75.0%", "≥ 88.9%"]
            })
            st.markdown("### Empirical Rule vs. Chebyshev's Inequality Evaluation")
            st.dataframe(df_rule, hide_index=True, use_container_width=True)
            st.caption("While Chebyshev's Inequality guarantees absolute mathematical bounds for any distribution, empirical stock returns closely align with the normal 68-95-99.7 rule.")


# ==========================================
# 5. SAMPLING & CLT
# ==========================================
elif nav_selection == "Sampling & CLT":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Sampling & Central Limit Theorem</div>
        <div class="page-desc">Compare samples with the full dataset and explore how sample means behave.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("Insufficient data available.")
    else:
        population = stock_return_series.to_numpy()
        N = len(population)
        pop_mean = float(np.mean(population))
        pop_std = float(np.std(population, ddof=1))
        
        tab_samp1, tab_samp2 = st.tabs(["Probability Sampling Methods", "Central Limit Theorem Simulation"])
        
        with tab_samp1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                n_sample = st.slider("Sample Size (n)", 10, min(150, N // 2), 30, 5)
            with col_c2:
                seed = st.number_input("Random Seed", 1, 9999, 42)
                
            # 1. SRS
            srs = stock_return_series.sample(n=n_sample, random_state=seed)
            # 2. Systematic
            k = max(1, N // n_sample)
            syst = stock_return_series.iloc[::k].head(n_sample)
            # 3. Stratified by Year
            df_yr = stock_return_series.rename("Return").to_frame()
            df_yr["Year"] = df_yr.index.year
            quota = max(1, int(np.ceil(n_sample / df_yr["Year"].nunique())))
            strat = (df_yr.groupby("Year", group_keys=False).apply(lambda g: g.sample(min(len(g), quota), random_state=seed)))["Return"].head(n_sample)
            
            samp_df = pd.DataFrame([
                {"Method": "Full Population (Ground Truth)", "Sample Size (n)": N, "Mean (%)": f"{pop_mean:.4f}%", "Std Dev (%)": f"{pop_std:.4f}%", "Difference from μ": "0.0000%"},
                {"Method": "Simple Random Sampling (SRS)", "Sample Size (n)": len(srs), "Mean (%)": f"{srs.mean():.4f}%", "Std Dev (%)": f"{srs.std(ddof=1):.4f}%", "Difference from μ": f"{abs(srs.mean()-pop_mean):.4f}%"},
                {"Method": f"Systematic Sampling (k={k})", "Sample Size (n)": len(syst), "Mean (%)": f"{syst.mean():.4f}%", "Std Dev (%)": f"{syst.std(ddof=1):.4f}%", "Difference from μ": f"{abs(syst.mean()-pop_mean):.4f}%"},
                {"Method": "Stratified Sampling (by Year)", "Sample Size (n)": len(strat), "Mean (%)": f"{strat.mean():.4f}%", "Std Dev (%)": f"{strat.std(ddof=1):.4f}%", "Difference from μ": f"{abs(strat.mean()-pop_mean):.4f}%"}
            ])
            st.dataframe(samp_df, hide_index=True, use_container_width=True)
            
        with tab_samp2:
            st.markdown("### Interactive Simulation of the Central Limit Theorem")
            st.caption("As sample size (n) increases, the distribution of sample averages approaches a normal distribution with standard error σ/√n.")
            
            col_clt1, col_clt2 = st.columns(2)
            with col_clt1:
                clt_n = st.select_slider("Sample Size (n)", options=[5, 15, 30, 50, 100], value=30)
            with col_clt2:
                clt_runs = st.select_slider("Simulation Draws (M)", options=[500, 1000, 2000], value=1000)
                
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
                st.metric("Mean of Sample Means", f"{mean_of_means:.4f}%", f"{mean_of_means - pop_mean:+.4f}%")
            with r3:
                st.metric("Theoretical Standard Error", f"{theoretical_se:.4f}%", f"Observed: {empirical_se:.4f}%")
                
            fig_clt = go.Figure()
            fig_clt.add_trace(go.Histogram(
                x=sample_means,
                histnorm='probability density',
                nbinsx=40,
                name="Simulated Means",
                marker_color='#3563E9',
                opacity=0.65
            ))
            
            x_vals = np.linspace(sample_means.min(), sample_means.max(), 150)
            clt_pdf = stats.norm.pdf(x_vals, loc=pop_mean, scale=theoretical_se)
            fig_clt.add_trace(go.Scatter(
                x=x_vals,
                y=clt_pdf,
                mode='lines',
                name="Theoretical Normal Curve N(μ, σ/√n)",
                line=dict(color='#F59E0B', width=2.2)
            ))
            fig_clt.add_vline(x=pop_mean, line_dash="dash", line_color="#EF4444", annotation_text="μ")
            
            fig_clt.update_layout(create_clean_layout(
                title=f"Sampling Distribution of Mean Daily Returns (n={clt_n}, M={clt_runs} draws)",
                xaxis_title="Sample Mean (%)",
                yaxis_title="Density",
                height=380
            ))
            st.plotly_chart(fig_clt, use_container_width=True)


# ==========================================
# 6. HYPOTHESIS TESTING
# ==========================================
elif nav_selection == "Hypothesis Testing":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Hypothesis Testing</div>
        <div class="page-desc">Test whether the observed results provide enough evidence to support a conclusion.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_return_series.empty:
        st.warning("Insufficient data available.")
    else:
        tab_h1, tab_h2, tab_h3, tab_h4 = st.tabs(["One-Sample & Z Tests", "Confidence Intervals", "Two-Sample & Paired Tests", "Chi-Square Tests"])
        
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
                t_concl = "Reject H0 (Statistically Significant)" if t_pval < alpha else "Fail to Reject H0 (Insignificant)"
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">One-Sample Student's t-Test</div>
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <strong>t-Statistic:</strong> {t_stat:.4f}<br>
                        <strong>Degrees of Freedom:</strong> {n_obs - 1}<br>
                        <strong>p-Value:</strong> {t_pval:.5f}<br>
                        <strong>Decision (α={alpha}):</strong> <span class="{'badge-pos' if t_pval < alpha else 'badge-neutral'}">{t_concl}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_th2:
                z_concl = "Reject H0" if z_pval < alpha else "Fail to Reject H0"
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Large-Sample Z Approximation</div>
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <strong>z-Statistic:</strong> {z_stat:.4f}<br>
                        <strong>Sample Size (n):</strong> {n_obs:,}<br>
                        <strong>Asymptotic p-Value:</strong> {z_pval:.5f}<br>
                        <strong>Decision (α={alpha}):</strong> <span class="{'badge-pos' if z_pval < alpha else 'badge-neutral'}">{z_concl}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.caption("Justification: When sample sizes exceed 1,000 observations, the Central Limit Theorem ensures that the sample mean distribution is asymptotically normal.")
            
        with tab_h2:
            st.markdown("### Two-Sided Confidence Intervals for Mean Daily Return")
            ci_rows = []
            for conf in [0.90, 0.95, 0.99]:
                a = 1 - conf
                t_crit = stats.t.ppf(1 - a / 2, df=n_obs - 1)
                moe = t_crit * se
                ci_rows.append({
                    "Confidence Level": f"{int(conf*100)}%",
                    "Critical Value (t*)": round(t_crit, 4),
                    "Margin of Error": f"±{moe:.4f}%",
                    "Lower Bound": f"{x_bar - moe:+.4f}%",
                    "Upper Bound": f"{x_bar + moe:+.4f}%",
                    "Contains Zero?": "Yes" if (x_bar - moe) <= 0 <= (x_bar + moe) else "No"
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
                    st.markdown(f"""
                    <div class="content-card">
                        <div class="card-title">Welch's Two-Sample t-Test</div>
                        <div style="font-size: 0.9rem; line-height: 1.8;">
                            <strong>t-Statistic:</strong> {welch.statistic:.4f}<br>
                            <strong>p-Value:</strong> {welch.pvalue:.5f}<br>
                            <strong>Result:</strong> {'Difference is statistically significant' if welch.pvalue < 0.05 else 'No significant difference in average return'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_w2:
                    st.markdown(f"""
                    <div class="content-card">
                        <div class="card-title">Paired t-Test (Matching Trading Days)</div>
                        <div style="font-size: 0.9rem; line-height: 1.8;">
                            <strong>t-Statistic:</strong> {rel.statistic:.4f}<br>
                            <strong>p-Value:</strong> {rel.pvalue:.5f}<br>
                            <strong>Synchronous Sessions:</strong> {len(paired):,}
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
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Direction Goodness-of-Fit Test</div>
                    <div style="font-size: 0.88rem; line-height: 1.7;">
                        H0: Equal 50/50 probability of positive vs. non-positive days.<br>
                        <strong>Positive Days:</strong> {pos_days} ({pos_days/n_obs*100:.1f}%)<br>
                        <strong>Non-Positive Days:</strong> {non_pos_days} ({non_pos_days/n_obs*100:.1f}%)<br>
                        <strong>Chi² Statistic:</strong> {chi2_gof:.4f}<br>
                        <strong>p-Value:</strong> {p_gof:.5f}<br>
                        <strong>Result:</strong> {'Evidence of directional bias' if p_gof < 0.05 else 'Consistent with symmetric 50/50 chance'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_ch2:
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Weekday Independence Test (Calendar Anomaly)</div>
                    <div style="font-size: 0.88rem; line-height: 1.7;">
                        H0: Return direction is independent of the day of the week.<br>
                        <strong>Chi² Statistic:</strong> {chi2_w:.4f}<br>
                        <strong>Degrees of Freedom:</strong> {dof_w}<br>
                        <strong>p-Value:</strong> {p_w:.5f}<br>
                        <strong>Result:</strong> {'Day-of-the-week pattern detected' if p_w < 0.05 else 'No statistically significant weekday effect'}
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
        <div class="page-desc">Explore how stock returns move together.</div>
    </div>
    """, unsafe_allow_html=True)
    
    corr_cols = [c for c in stock_tickers if c in active_returns.columns]
    if use_benchmark and BENCHMARK in active_returns.columns:
        corr_cols.append(BENCHMARK)
    df_corr = active_returns[corr_cols].dropna()
    
    if df_corr.empty or len(corr_cols) < 2:
        st.warning("Insufficient multi-asset data.")
    else:
        tab_cr1, tab_cr2, tab_cr3 = st.tabs(["Correlation Matrices", "Stock vs. Benchmark Scatter (Beta)", "Pairwise Stock Scatter"])
        
        with tab_cr1:
            m_type = st.radio("Metric", ["Pearson Correlation (Linear)", "Spearman Correlation (Rank)", "Covariance"], horizontal=True)
            
            if "Pearson" in m_type:
                mat = df_corr.corr(method="pearson")
                c_title = "Pearson Linear Correlation Matrix"
            elif "Spearman" in m_type:
                mat = df_corr.corr(method="spearman")
                c_title = "Spearman Rank Correlation Matrix"
            else:
                mat = df_corr.cov()
                c_title = "Covariance Matrix"
                
            fig_hm = go.Figure(data=go.Heatmap(
                z=mat.values,
                x=mat.columns,
                y=mat.index,
                colorscale="Blues",
                zmin=-1 if "Covariance" not in m_type else None,
                zmax=1 if "Covariance" not in m_type else None,
                text=np.round(mat.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=11, family="Inter, sans-serif")
            ))
            fig_hm.update_layout(create_clean_layout(
                title=c_title,
                height=420
            ))
            st.plotly_chart(fig_hm, use_container_width=True)
            
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
                    st.metric("Beta (β - Systematic Risk)", f"{slope:.3f}", help="Sensitivity to broader market movements.")
                with col_b2:
                    st.metric("Alpha (α - Daily)", f"{intercept:+.4f}%", help="Intercept term.")
                with col_b3:
                    st.metric("R² (Variance Explained)", f"{r_val**2:.3f}", f"Correlation: {r_val:.3f}")
                    
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=x_val,
                    y=y_val,
                    mode='markers',
                    name="Daily Sessions",
                    marker=dict(color='#3563E9', size=5, opacity=0.45)
                ))
                
                x_line = np.linspace(x_val.min(), x_val.max(), 100)
                fig_sc.add_trace(go.Scatter(
                    x=x_line,
                    y=intercept + slope * x_line,
                    mode='lines',
                    name=f"OLS Regression (β={slope:.2f})",
                    line=dict(color='#F59E0B', width=2)
                ))
                
                fig_sc.update_layout(create_clean_layout(
                    title=f"{selected_stock} vs. S&P 500 Daily Return Scatter & OLS Line",
                    xaxis_title="S&P 500 Daily Return (%)",
                    yaxis_title=f"{selected_stock} Daily Return (%)",
                    height=400
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
                    marker=dict(color='#3563E9', size=5, opacity=0.45)
                ))
                fig_p.update_layout(create_clean_layout(
                    title=f"Return Scatter: {selected_stock} vs. {pair_target} (r = {pr_r:.3f}, p = {pr_p:.1e})",
                    xaxis_title=f"{pair_target} Daily Return (%)",
                    yaxis_title=f"{selected_stock} Daily Return (%)",
                    height=380
                ))
                st.plotly_chart(fig_p, use_container_width=True)


# ==========================================
# 8. MODEL PERFORMANCE
# ==========================================
elif nav_selection == "Model Performance":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model Performance</div>
        <div class="page-desc">Compare the performance of the trained prediction models.</div>
    </div>
    """, unsafe_allow_html=True)
    
    df_reg = data_dict.get("regression_comparison")
    df_clf = data_dict.get("classification_comparison")
    df_cm = data_dict.get("confusion_matrix")
    
    tab_mp1, tab_mp2, tab_mp3 = st.tabs(["Regression Models", "Classification Models", "Holdout Confusion Matrix"])
    
    with tab_mp1:
        if df_reg is not None:
            st.markdown("### Next-Day Return Regression Leaderboard")
            st.caption("Target: Next-day percentage return. Evaluated on chronological 20% holdout test set:")
            st.dataframe(df_reg.round(4), hide_index=True, use_container_width=True)
            
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(x=df_reg["Model"], y=df_reg["RMSE"], name="RMSE", marker_color='#3563E9'))
            fig_reg.add_trace(go.Bar(x=df_reg["Model"], y=df_reg["MAE"], name="MAE", marker_color='#94A3B8'))
            fig_reg.update_layout(create_clean_layout(
                title="Model Error Metrics (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="Error (%)",
                height=350
            ))
            st.plotly_chart(fig_reg, use_container_width=True)
            
            with st.expander("Why are R² scores close to zero?"):
                st.caption("Consistent with the Efficient Market Hypothesis, short-term daily asset returns exhibit low signal-to-noise ratios. Linear models capture aggregate trends, but next-day percentage returns cannot be predicted with high precision from simple historical lags.")
                
    with tab_mp2:
        if df_clf is not None:
            st.markdown("### Market Direction Classification Leaderboard")
            st.caption("Target: Next-day return > 0 (1 = Up, 0 = Flat/Down). Evaluated on 20% holdout test set:")
            st.dataframe(df_clf.round(4), hide_index=True, use_container_width=True)
            
            fig_clf = go.Figure()
            fig_clf.add_trace(go.Bar(x=df_clf["Model"], y=df_clf["Accuracy"], name="Accuracy", marker_color='#3563E9'))
            fig_clf.add_trace(go.Bar(x=df_clf["Model"], y=df_clf["F1-Score"], name="F1-Score", marker_color='#10B981'))
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
                    height=340
                ))
                st.plotly_chart(fig_cmat, use_container_width=True)
                
            with col_m2:
                tn = df_cm.iloc[0, 0]
                fp = df_cm.iloc[0, 1]
                fn = df_cm.iloc[1, 0]
                tp = df_cm.iloc[1, 1]
                tot = tn + fp + fn + tp
                
                st.markdown(f"""
                <div class="content-card">
                    <div class="card-title">Diagnostic Breakdown</div>
                    <div style="font-size: 0.88rem; line-height: 1.8;">
                        • <strong>Test Observations:</strong> {tot} sessions (chronological holdout)<br>
                        • <strong>True Negatives:</strong> {tn} (Correctly classified down days)<br>
                        • <strong>False Positives:</strong> {fp} (Type I error)<br>
                        • <strong>False Negatives:</strong> {fn} (Type II error)<br>
                        • <strong>True Positives:</strong> {tp} (Correctly classified up days)<br>
                        • <strong>Accuracy:</strong> {(tn+tp)/tot*100:.2f}%<br>
                        • <strong>Precision:</strong> {tp/(tp+fp)*100:.2f}%<br>
                        • <strong>Sensitivity (Recall):</strong> {tp/(tp+fn)*100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# 9. PREDICTIONS
# ==========================================
elif nav_selection == "Predictions":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predictions</div>
        <div class="page-desc">Explore the estimates produced by the saved models.</div>
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
        
        st.markdown(f"""
        <div class="content-card" style="padding: 14px 18px; margin-bottom: 16px;">
            <div style="font-size: 0.88rem; color: #475569;">
                <strong>Pipeline Configuration:</strong> The saved models (<strong>Lasso Regression</strong> and <strong>Logistic Regression</strong>) 
                were trained on <strong>Apple (AAPL)</strong> historical features with S&P 500 benchmark inputs across 8 technical dimensions: 
                <code>Return_Lag1, Return_Lag2, Return_Lag5, Volatility_5D, Volatility_20D, Moving_Average_5D, Moving_Average_20D, Benchmark_Return</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        p_mode = st.radio("Inference Input Source:", ["Latest Historical Features (Real Data)", "Custom Scenario Simulator"], horizontal=True)
        
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
                st.markdown(f"#### Active Feature Values as of {ref_date}")
                st.dataframe(feat_df.round(4), hide_index=True, use_container_width=True)
                
        else:
            st.markdown("#### Adjust Market Parameters")
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
                st.markdown("### Model Estimates")
                
                col_o1, col_o2, col_o3 = st.columns(3)
                with col_o1:
                    r_sign = "+" if pred_ret >= 0 else ""
                    b_color = "badge-pos" if pred_ret >= 0 else "badge-neg"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Next-Day Return</div>
                        <div class="metric-value">{r_sign}{pred_ret:.4f}%</div>
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
                        <div class="metric-label">Direction Confidence</div>
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
            trained on historical equity data solely for university coursework and demonstration purposes. 
            Financial markets are non-stationary and subject to complex macroeconomic shocks. 
            <strong>Under no circumstances should these estimates be construed as financial advice, investment recommendations, or trading signals.</strong>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("<hr style='margin: 32px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.8rem; padding-bottom: 12px;">
    <strong>Stock Market Return Statistical Analyzer</strong> • Academic Analytics Dashboard
</div>
""", unsafe_allow_html=True)

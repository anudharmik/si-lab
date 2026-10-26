"""
Stock Market Return Statistical Analyzer
A comprehensive, professional Streamlit application for financial return analytics,
classical inferential statistics, sampling theory, and machine learning evaluation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Stock Market Return Statistical Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern financial UI
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .metric-sub.positive { color: #34d399; }
    .metric-sub.negative { color: #f87171; }
    .metric-sub.neutral { color: #94a3b8; }
    
    /* Section Containers */
    .section-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 24px;
    }
    .section-title {
        color: #38bdf8;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-primary { background: #0369a1; color: #e0f2fe; }
    .badge-success { background: #065f46; color: #d1fae5; }
    .badge-warning { background: #854d0e; color: #fef9c3; }
    .badge-danger { background: #991b1b; color: #fee2e2; }
    
    /* Callout & Disclaimer */
    .callout-info {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 18px;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .callout-disclaimer {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 24px;
        margin-bottom: 24px;
        color: #fca5a5;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MATPLOTLIB STYLING HELPER
# ==========================================
def apply_plot_style():
    """Sets a clean, dark-themed aesthetic for matplotlib and seaborn figures."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0f172a',
        'axes.facecolor': '#1e293b',
        'axes.edgecolor': '#334155',
        'axes.labelcolor': '#cbd5e1',
        'axes.grid': True,
        'grid.color': '#334155',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'text.color': '#f8fafc',
        'legend.facecolor': '#1e293b',
        'legend.edgecolor': '#334155',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.autolayout': True
    })


# ==========================================
# DATA & MODEL LOADING WITH CACHING
# ==========================================
APP_DIR = Path(__file__).resolve().parent

def resolve_file(rel_path: str):
    """Checks possible locations for a data or model file with fallbacks."""
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
    """Loads all CSV datasets, cleans dates, and prepares structured dataframes."""
    data = {}
    errors = []

    # 1. Close prices
    p_close = resolve_file("stock_analyzer_data/stock_close_prices.csv")
    if p_close:
        try:
            df_close = pd.read_csv(p_close)
            date_col = "Date" if "Date" in df_close.columns else df_close.columns[0]
            df_close[date_col] = pd.to_datetime(df_close[date_col])
            df_close = df_close.sort_values(date_col).reset_index(drop=True)
            df_close.set_index(date_col, inplace=True)
            data["close_prices"] = df_close
        except Exception as e:
            errors.append(f"Error parsing stock_close_prices.csv: {e}")
    else:
        errors.append("stock_close_prices.csv not found in expected directories.")

    # 2. Daily returns
    p_ret = resolve_file("stock_analyzer_data/daily_stock_returns.csv")
    if p_ret:
        try:
            df_ret = pd.read_csv(p_ret)
            date_col = "index" if "index" in df_ret.columns else ("Date" if "Date" in df_ret.columns else df_ret.columns[0])
            df_ret[date_col] = pd.to_datetime(df_ret[date_col])
            df_ret = df_ret.sort_values(date_col).reset_index(drop=True)
            df_ret.set_index(date_col, inplace=True)
            data["returns"] = df_ret
        except Exception as e:
            errors.append(f"Error parsing daily_stock_returns.csv: {e}")
    else:
        errors.append("daily_stock_returns.csv not found in expected directories.")

    # 3. Modeling dataset
    p_model = resolve_file("stock_analyzer_data/modeling_dataset.csv")
    if p_model:
        try:
            df_model = pd.read_csv(p_model)
            if "Date" in df_model.columns:
                df_model["Date"] = pd.to_datetime(df_model["Date"])
                df_model = df_model.sort_values("Date").reset_index(drop=True)
                df_model.set_index("Date", inplace=True)
            data["modeling_dataset"] = df_model
        except Exception as e:
            errors.append(f"Error parsing modeling_dataset.csv: {e}")
    else:
        errors.append("modeling_dataset.csv not found in expected directories.")

    # 4. Regression comparison
    p_reg = resolve_file("stock_analyzer_data/regression_model_comparison.csv")
    if p_reg:
        try:
            data["regression_comparison"] = pd.read_csv(p_reg)
        except Exception as e:
            errors.append(f"Error parsing regression_model_comparison.csv: {e}")

    # 5. Classification comparison
    p_clf = resolve_file("stock_analyzer_data/classification_model_comparison.csv")
    if p_clf:
        try:
            data["classification_comparison"] = pd.read_csv(p_clf)
        except Exception as e:
            errors.append(f"Error parsing classification_model_comparison.csv: {e}")

    # 6. Confusion matrix
    p_cm = resolve_file("stock_analyzer_data/classification_confusion_matrix.csv")
    if p_cm:
        try:
            data["confusion_matrix"] = pd.read_csv(p_cm, index_col=0)
        except Exception as e:
            errors.append(f"Error parsing classification_confusion_matrix.csv: {e}")

    # 7. S&P 500 Ticker List (Optional reference)
    p_sp = resolve_file("stock_analyzer_data/sp500_ticker_list.csv")
    if p_sp:
        try:
            data["sp500_tickers"] = pd.read_csv(p_sp)
        except Exception:
            pass

    return data, errors

@st.cache_resource(show_spinner=False)
def load_models():
    """Loads fitted regression and classification pipeline models with cross-version patch."""
    models = {}
    errors = []

    p_reg = resolve_file("models/best_return_regression_model.pkl")
    p_clf = resolve_file("models/best_direction_classification_model.pkl")

    if p_reg:
        try:
            reg = joblib.load(p_reg)
            # Patch cross-version SimpleImputer attribute change in scikit-learn
            imp = getattr(reg, "named_steps", {}).get("imputer")
            if imp is not None and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = getattr(imp, "_fit_dtype", float)
            models["regression"] = reg
        except Exception as e:
            errors.append(f"Failed to load regression model ({p_reg}): {e}")
    else:
        errors.append("best_return_regression_model.pkl not found in models/ or scratch/.")

    if p_clf:
        try:
            clf = joblib.load(p_clf)
            # Patch cross-version SimpleImputer attribute change in scikit-learn
            imp = getattr(clf, "named_steps", {}).get("imputer")
            if imp is not None and not hasattr(imp, "_fill_dtype"):
                imp._fill_dtype = getattr(imp, "_fit_dtype", float)
            models["classification"] = clf
        except Exception as e:
            errors.append(f"Failed to load classification model ({p_clf}): {e}")
    else:
        errors.append("best_direction_classification_model.pkl not found in models/ or scratch/.")

    return models, errors


# ==========================================
# STATISTICAL CALCULATION UTILITIES
# ==========================================
def calculate_descriptive_stats(series: pd.Series) -> pd.Series:
    """Computes exact statistical moments matching the project notebook."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    
    q1 = float(clean.quantile(0.25))
    q2 = float(clean.quantile(0.50))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    
    # Mode calculation
    mode_vals = clean.round(4).mode()
    mode_val = float(mode_vals.iloc[0]) if not mode_vals.empty else float(clean.iloc[0])
    
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
        "IQR (%)": iqr,
        "Skewness": float(clean.skew()),
        "Kurtosis (Fisher)": float(clean.kurtosis())
    })


# ==========================================
# INITIAL DATA ACQUISITION & VALIDATION
# ==========================================
data_dict, data_errors = load_data()
model_dict, model_errors = load_models()

# Apply plot theme
apply_plot_style()

# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1>Stock Market Return Statistical Analyzer</h1>
            <p>Empirical Quantitative Modeling & Statistical Analysis for Mega-Cap Equities vs. S&P 500</p>
        </div>
        <div>
            <span class="badge badge-primary">College Project Edition</span>
            <span class="badge badge-success" style="margin-left: 6px;">Python 3.14 & SciPy & Scikit-Learn</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Missing Data Diagnostic Check
if "returns" not in data_dict or "close_prices" not in data_dict:
    st.error("🚨 Critical files could not be loaded. Please ensure dataset CSVs are placed in `stock_analyzer_data/` or `scratch/`.")
    if data_errors:
        st.write("### Error Details:")
        for err in data_errors:
            st.write(f"- {err}")
    st.stop()

df_prices = data_dict["close_prices"]
df_returns = data_dict["returns"]
df_model_data = data_dict.get("modeling_dataset", None)

# Identify available equity tickers & benchmark
all_tickers = [c for c in df_returns.columns if c not in ["Date", "index"]]
BENCHMARK = "^GSPC"
stock_tickers = [t for t in all_tickers if t != BENCHMARK]

if not stock_tickers:
    stock_tickers = all_tickers

DEFAULT_TICKER = "AAPL" if "AAPL" in stock_tickers else stock_tickers[0]


# ==========================================
# SIDEBAR CONTROLS & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Parameters")
    
    # 1. Primary Stock Selector
    selected_stock = st.selectbox(
        "Select Target Stock",
        options=stock_tickers,
        index=stock_tickers.index(DEFAULT_TICKER) if DEFAULT_TICKER in stock_tickers else 0,
        help="Select primary equity asset for statistical calculations and distributions."
    )
    
    # 2. Benchmark Selector
    has_benchmark = BENCHMARK in df_returns.columns
    use_benchmark = st.checkbox(
        f"Include Benchmark ({BENCHMARK})",
        value=has_benchmark,
        disabled=not has_benchmark,
        help="Overlay S&P 500 (^GSPC) benchmark where available."
    )
    
    st.markdown("---")
    
    # 3. Date Range Filter
    min_date = df_returns.index.min().to_pydatetime()
    max_date = df_returns.index.max().to_pydatetime()
    
    st.markdown("📅 **Filter Date Window**")
    date_preset = st.radio(
        "Quick Preset",
        ["All Data (5 Years)", "Past 3 Years", "Past 1 Year", "Custom Range"],
        index=0,
        horizontal=False
    )
    
    if date_preset == "Past 1 Year":
        filter_start = max(min_date, max_date - pd.Timedelta(days=365))
        filter_end = max_date
    elif date_preset == "Past 3 Years":
        filter_start = max(min_date, max_date - pd.Timedelta(days=365 * 3))
        filter_end = max_date
    elif date_preset == "Custom Range":
        filter_range = st.date_input(
            "Custom Window",
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
    
    # Convert to pandas Timestamps
    ts_start = pd.to_datetime(filter_start)
    ts_end = pd.to_datetime(filter_end)
    
    st.caption(f"Active window: `{ts_start.strftime('%Y-%m-%d')}` to `{ts_end.strftime('%Y-%m-%d')}`")
    st.markdown("---")
    
    # 4. Navigation Menu
    st.markdown("### 🧭 Dashboard Sections")
    nav_selection = st.radio(
        "Navigate to:",
        [
            "1. Executive Overview",
            "2. Historical Analysis",
            "3. Statistical Analysis",
            "4. Distribution Analysis",
            "5. Sampling & Probability (CLT)",
            "6. Hypothesis Testing",
            "7. Correlation Analysis",
            "8. Model Evaluation",
            "9. Predictive Inference"
        ],
        index=0
    )

# Slice active data according to selected date range
active_returns = df_returns.loc[(df_returns.index >= ts_start) & (df_returns.index <= ts_end)].copy()
active_prices = df_prices.loc[(df_prices.index >= ts_start) & (df_prices.index <= ts_end)].copy()

stock_return_series = active_returns[selected_stock].dropna()
stock_price_series = active_prices[selected_stock].dropna() if selected_stock in active_prices.columns else pd.Series()
benchmark_return_series = active_returns[BENCHMARK].dropna() if (use_benchmark and BENCHMARK in active_returns.columns) else None


# ==========================================
# 1. OVERVIEW
# ==========================================
if nav_selection == "1. Executive Overview":
    st.markdown("## 📊 Executive Overview")
    st.markdown(f"High-level market and statistical indicators for **{selected_stock}** across the selected observation window.")
    
    if stock_return_series.empty or stock_price_series.empty:
        st.warning("No data points available in the selected date range.")
    else:
        # Key Financial Statistics Computations
        latest_date = stock_price_series.index.max().strftime('%Y-%m-%d')
        latest_close = stock_price_series.iloc[-1]
        
        # Cumulative return over window: (P_end - P_start) / P_start * 100
        p_start = stock_price_series.iloc[0]
        p_end = stock_price_series.iloc[-1]
        cumulative_return = ((p_end - p_start) / p_start) * 100
        
        avg_daily_return = stock_return_series.mean()
        daily_volatility = stock_return_series.std(ddof=1)
        annualized_return = avg_daily_return * 252
        annualized_volatility = daily_volatility * np.sqrt(252)
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else np.nan
        
        # Maximum Drawdown calculation
        rolling_max = stock_price_series.cummax()
        drawdown_series = (stock_price_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown_series.min()
        
        # KPI Cards (Top Row)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Selected Equity</div>
                <div class="metric-value">{selected_stock}</div>
                <div class="metric-sub neutral">Latest Date: {latest_date}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Close Price</div>
                <div class="metric-value">${latest_close:,.2f}</div>
                <div class="metric-sub neutral">Base: ${p_start:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            cum_class = "positive" if cumulative_return >= 0 else "negative"
            cum_sign = "+" if cumulative_return >= 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cumulative Return</div>
                <div class="metric-value" style="color: {'#34d399' if cumulative_return >= 0 else '#f87171'}">{cum_sign}{cumulative_return:.2f}%</div>
                <div class="metric-sub {cum_class}">{len(stock_return_series)} Trading Days</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Volatility (σ)</div>
                <div class="metric-value">{daily_volatility:.2f}%</div>
                <div class="metric-sub neutral">Ann. Vol: {annualized_volatility:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # KPI Cards (Bottom Row)
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            m_sign = "+" if avg_daily_return >= 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Daily Return</div>
                <div class="metric-value">{m_sign}{avg_daily_return:.4f}%</div>
                <div class="metric-sub neutral">Ann. Return: {m_sign}{annualized_return:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio (Rf=0%)</div>
                <div class="metric-value">{sharpe_ratio:.2f}</div>
                <div class="metric-sub neutral">Risk-Adjusted Performance</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Maximum Drawdown</div>
                <div class="metric-value" style="color: #f87171;">{max_drawdown:.2f}%</div>
                <div class="metric-sub neutral">Peak-to-Trough Decline</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col8:
            pos_days = (stock_return_series > 0).sum()
            win_rate = (pos_days / len(stock_return_series)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positive Days Ratio</div>
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-sub neutral">{pos_days} Up / {len(stock_return_series) - pos_days} Down</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Normalized Cumulative Growth vs Benchmark Preview
        st.markdown("### 📈 Cumulative Performance Comparison (Base = 100)")
        
        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
        norm_stock = (stock_price_series / stock_price_series.iloc[0]) * 100
        ax.plot(norm_stock.index, norm_stock.values, label=f"{selected_stock} (+{cumulative_return:.1f}%)", color="#38bdf8", linewidth=2.2)
        
        if benchmark_return_series is not None and not benchmark_return_series.empty and BENCHMARK in active_prices.columns:
            bm_prices = active_prices[BENCHMARK].dropna()
            norm_bm = (bm_prices / bm_prices.iloc[0]) * 100
            bm_cum = ((bm_prices.iloc[-1] - bm_prices.iloc[0]) / bm_prices.iloc[0]) * 100
            ax.plot(norm_bm.index, norm_bm.values, label=f"S&P 500 (^GSPC) (+{bm_cum:.1f}%)", color="#94a3b8", linestyle="--", linewidth=1.8)
            
        ax.set_title(f"Rebased Historical Asset Growth: {selected_stock} vs. Benchmark", fontsize=13, fontweight=600, pad=12)
        ax.set_ylabel("Normalized Growth (Index = 100)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.legend(loc="upper left", frameon=True)
        st.pyplot(fig)
        plt.close(fig)
        
        # Summary Table
        st.markdown("### 📋 Multi-Asset Summary Snapshot")
        summary_rows = []
        for t in stock_tickers:
            if t in active_returns.columns and t in active_prices.columns:
                ret_s = active_returns[t].dropna()
                prc_s = active_prices[t].dropna()
                if not ret_s.empty and not prc_s.empty:
                    c_ret = ((prc_s.iloc[-1] - prc_s.iloc[0]) / prc_s.iloc[0]) * 100
                    d_vol = ret_s.std(ddof=1)
                    ann_vol = d_vol * np.sqrt(252)
                    summary_rows.append({
                        "Ticker": t,
                        "Latest Close ($)": round(prc_s.iloc[-1], 2),
                        "Cumulative Return (%)": round(c_ret, 2),
                        "Avg Daily Return (%)": round(ret_s.mean(), 4),
                        "Daily Volatility (%)": round(d_vol, 2),
                        "Ann. Volatility (%)": round(ann_vol, 2),
                        "Sharpe Ratio": round((ret_s.mean() * 252) / ann_vol, 2) if ann_vol > 0 else np.nan
                    })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows).set_index("Ticker"), use_container_width=True)


# ==========================================
# 2. HISTORICAL ANALYSIS
# ==========================================
elif nav_selection == "2. Historical Analysis":
    st.markdown("## 📈 Historical Price & Return Trajectory")
    st.markdown(f"In-depth inspection of historical pricing, moving averages, drawdowns, and wealth accumulation for **{selected_stock}**.")
    
    if stock_price_series.empty or stock_return_series.empty:
        st.warning("Insufficient data available in the selected window.")
    else:
        tab_p1, tab_p2, tab_p3 = st.tabs(["Closing Price & Moving Averages", "Wealth Growth ($10,000 Invested)", "Daily Return Fluctuations & Drawdown"])
        
        with tab_p1:
            # Price + 20D & 50D Moving Averages
            ma20 = stock_price_series.rolling(window=20).mean()
            ma50 = stock_price_series.rolling(window=50).mean()
            
            fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
            ax.plot(stock_price_series.index, stock_price_series.values, label=f"{selected_stock} Close", color="#38bdf8", linewidth=2)
            ax.plot(ma20.index, ma20.values, label="20-Day Moving Average", color="#f59e0b", linestyle="-.", linewidth=1.4)
            ax.plot(ma50.index, ma50.values, label="50-Day Moving Average", color="#a855f7", linestyle=":", linewidth=1.4)
            
            ax.set_title(f"{selected_stock} Historical Adjusted Closing Price with Moving Averages", fontsize=13, fontweight=600, pad=12)
            ax.set_ylabel("Price ($ USD)", fontsize=10)
            ax.set_xlabel("Date", fontsize=10)
            ax.legend(loc="upper left")
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption("Moving averages smooth out high-frequency market noise to illustrate structural intermediate (20D) and longer-term (50D) trends.")
            
        with tab_p2:
            # Growth of $10,000 initial investment
            growth_stock = (1 + stock_return_series / 100).cumprod() * 10000
            
            fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
            ax.plot(growth_stock.index, growth_stock.values, label=f"{selected_stock} Portfolio", color="#10b981", linewidth=2.2)
            
            if benchmark_return_series is not None and not benchmark_return_series.empty:
                growth_bm = (1 + benchmark_return_series / 100).cumprod() * 10000
                ax.plot(growth_bm.index, growth_bm.values, label="S&P 500 (^GSPC) Portfolio", color="#94a3b8", linestyle="--", linewidth=1.8)
                
            ax.axhline(10000, color="#64748b", linestyle=":", label="Initial Capital ($10,000)")
            ax.set_title("Growth of an Initial $10,000 Investment", fontsize=13, fontweight=600, pad=12)
            ax.set_ylabel("Portfolio Value ($ USD)", fontsize=10)
            ax.set_xlabel("Date", fontsize=10)
            ax.legend(loc="upper left")
            st.pyplot(fig)
            plt.close(fig)
            
            final_val = growth_stock.iloc[-1]
            st.info(f"An initial investment of **$10,000** in **{selected_stock}** on `{stock_return_series.index.min().strftime('%Y-%m-%d')}` would have evolved to **${final_val:,.2f}** by `{stock_return_series.index.max().strftime('%Y-%m-%d')}`.")
            
        with tab_p3:
            # Daily Returns Bar Chart + Drawdown Chart
            fig, (ax_ret, ax_dd) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=120, gridspec_kw={'height_ratios': [2, 1.2]})
            
            colors = np.where(stock_return_series >= 0, '#34d399', '#f87171')
            ax_ret.bar(stock_return_series.index, stock_return_series.values, color=colors, width=1.5, alpha=0.85)
            ax_ret.axhline(0, color="#64748b", linewidth=0.8)
            ax_ret.set_title(f"{selected_stock} Daily Percentage Returns and Underwater Drawdown", fontsize=13, fontweight=600, pad=12)
            ax_ret.set_ylabel("Daily Return (%)", fontsize=10)
            
            # Drawdown
            rolling_max = stock_price_series.cummax()
            dd = (stock_price_series - rolling_max) / rolling_max * 100
            ax_dd.plot(dd.index, dd.values, color="#f87171", linewidth=1.5)
            ax_dd.fill_between(dd.index, dd.values, 0, color="#ef4444", alpha=0.3)
            ax_dd.set_ylabel("Drawdown (%)", fontsize=10)
            ax_dd.set_xlabel("Date", fontsize=10)
            
            st.pyplot(fig)
            plt.close(fig)


# ==========================================
# 3. STATISTICAL ANALYSIS
# ==========================================
elif nav_selection == "3. Statistical Analysis":
    st.markdown("## 🔢 Classical Descriptive Statistics")
    st.markdown(f"Parametric and non-parametric statistical metrics computed from daily percentage returns for **{selected_stock}**.")
    
    if stock_return_series.empty:
        st.warning("No data points available for statistical analysis.")
    else:
        stats_series = calculate_descriptive_stats(stock_return_series)
        
        col_st1, col_st2 = st.columns([1, 1.2])
        
        with col_st1:
            st.markdown(f"### 📋 Statistical Summary: {selected_stock}")
            df_display = pd.DataFrame({
                "Metric": stats_series.index,
                "Value": [f"{v:.6f}" if isinstance(v, float) else str(v) for v in stats_series.values]
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
        with col_st2:
            st.markdown("### 💡 Statistical Interpretation")
            mean_v = stats_series["Mean (%)"]
            median_v = stats_series["Median (%)"]
            skew_v = stats_series["Skewness"]
            kurt_v = stats_series["Kurtosis (Fisher)"]
            std_v = stats_series["Std Deviation (%)"]
            
            skew_desc = "positively skewed (right-tailed)" if skew_v > 0.1 else ("negatively skewed (left-tailed)" if skew_v < -0.1 else "approximately symmetric")
            kurt_desc = "leptokurtic (fat-tailed, higher risk of extreme outliers)" if kurt_v > 0 else "platykurtic (thinner tails than normal)"
            
            st.markdown(f"""
            <div class="callout-info">
                <strong>Central Tendency:</strong><br>
                • <strong>Sample Mean:</strong> {mean_v:+.4f}% daily return.<br>
                • <strong>Sample Median:</strong> {median_v:+.4f}% daily return.<br>
                • Because mean {'<' if mean_v < median_v else '>'} median, the distribution exhibits a {skew_desc} tendency.<br><br>
                <strong>Dispersion & Volatility:</strong><br>
                • <strong>Standard Deviation (s):</strong> {std_v:.4f}% daily variability.<br>
                • <strong>Interquartile Range (IQR):</strong> {stats_series['IQR (%)']:.4f}% covers the central 50% of observed trading returns.<br><br>
                <strong>Higher-Order Moments (Risk Profile):</strong><br>
                • <strong>Sample Skewness:</strong> {skew_v:.4f} ({skew_desc}).<br>
                • <strong>Excess Kurtosis:</strong> {kurt_v:.4f} ({kurt_desc}). Financial returns typically display excess kurtosis due to sudden news shocks and volatility clustering.
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Multi-Stock Comparative Matrix
        st.markdown("### 📊 Cross-Stock Statistical Comparison Table")
        multi_stats = {}
        for t in stock_tickers:
            if t in active_returns.columns:
                s = active_returns[t].dropna()
                if not s.empty:
                    multi_stats[t] = calculate_descriptive_stats(s)
        if use_benchmark and BENCHMARK in active_returns.columns:
            bm_s = active_returns[BENCHMARK].dropna()
            if not bm_s.empty:
                multi_stats[BENCHMARK] = calculate_descriptive_stats(bm_s)
                
        if multi_stats:
            df_multi = pd.DataFrame(multi_stats).T
            st.dataframe(df_multi.round(4), use_container_width=True)


# ==========================================
# 4. DISTRIBUTION ANALYSIS
# ==========================================
elif nav_selection == "4. Distribution Analysis":
    st.markdown("## 📉 Distribution & Density Analysis")
    st.markdown(f"Empirical probability density, Gaussian normality comparison, and extreme tail assessment for **{selected_stock}**.")
    
    if stock_return_series.empty:
        st.warning("No data points available for distribution analysis.")
    else:
        clean_ret = stock_return_series.to_numpy()
        mu = np.mean(clean_ret)
        sigma = np.std(clean_ret, ddof=1)
        
        tab_d1, tab_d2, tab_d3 = st.tabs(["Histogram & Normal Curve Fit", "Box Plot & Outlier Detection", "Empirical vs. Chebyshev Bounds"])
        
        with tab_d1:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
            
            # Histogram
            count, bins, _ = ax.hist(clean_ret, bins=45, density=True, alpha=0.6, color="#0284c7", edgecolor="#0f172a", label="Empirical Density")
            
            # Kernel Density Estimate (KDE)
            try:
                sns.kdeplot(clean_ret, ax=ax, color="#38bdf8", linewidth=2.2, label="Kernel Density Estimate (KDE)")
            except Exception:
                pass
                
            # Fitted Normal Curve
            x_range = np.linspace(clean_ret.min() - 0.5, clean_ret.max() + 0.5, 300)
            normal_pdf = stats.norm.pdf(x_range, loc=mu, scale=sigma)
            ax.plot(x_range, normal_pdf, color="#f59e0b", linestyle="--", linewidth=2, label=f"Theoretical Normal Fit (μ={mu:.2f}%, σ={sigma:.2f}%)")
            
            ax.axvline(mu, color="#ef4444", linestyle=":", linewidth=1.5, label=f"Mean Return ({mu:.2f}%)")
            ax.axvline(np.median(clean_ret), color="#10b981", linestyle=":", linewidth=1.5, label=f"Median Return ({np.median(clean_ret):.2f}%)")
            
            ax.set_title(f"Empirical Daily Return Distribution vs. Normal Benchmark: {selected_stock}", fontsize=13, fontweight=600, pad=12)
            ax.set_xlabel("Daily Return (%)", fontsize=10)
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption("Notice the sharper peak (leptokurtosis) and heavier tails compared to the Gaussian bell curve. This represents classic financial market behavior.")
            
        with tab_d2:
            fig, (ax_box1, ax_box2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
            
            # Single stock boxplot
            sns.boxplot(x=clean_ret, ax=ax_box1, color="#0284c7", flierprops={"marker": "o", "color": "#f87171", "alpha": 0.5})
            ax_box1.set_title(f"{selected_stock} Box Plot (Outlier Detection)", fontsize=11, fontweight=600)
            ax_box1.set_xlabel("Daily Return (%)", fontsize=10)
            
            # Comparative boxplots
            compare_data = []
            compare_labels = []
            for t in stock_tickers:
                if t in active_returns.columns:
                    s = active_returns[t].dropna()
                    if not s.empty:
                        compare_data.append(s)
                        compare_labels.append(t)
            if compare_data:
                sns.boxplot(data=compare_data, ax=ax_box2, palette="Blues")
                ax_box2.set_xticklabels(compare_labels, rotation=30)
                ax_box2.set_title("Cross-Asset Return Dispersion", fontsize=11, fontweight=600)
                ax_box2.set_ylabel("Daily Return (%)", fontsize=10)
                
            st.pyplot(fig)
            plt.close(fig)
            
        with tab_d3:
            # Empirical Rule (68-95-99.7) vs Chebyshev's Inequality
            n_tot = len(clean_ret)
            within_1s = np.sum((clean_ret >= mu - sigma) & (clean_ret <= mu + sigma)) / n_tot * 100
            within_2s = np.sum((clean_ret >= mu - 2*sigma) & (clean_ret <= mu + 2*sigma)) / n_tot * 100
            within_3s = np.sum((clean_ret >= mu - 3*sigma) & (clean_ret <= mu + 3*sigma)) / n_tot * 100
            
            bounds_table = pd.DataFrame({
                "Range Interval": ["Within ±1σ", "Within ±2σ", "Within ±3σ"],
                "Absolute Return Range (%)": [
                    f"[{mu - sigma:.2f}%, {mu + sigma:.2f}%]",
                    f"[{mu - 2*sigma:.2f}%, {mu + 2*sigma:.2f}%]",
                    f"[{mu - 3*sigma:.2f}%, {mu + 3*sigma:.2f}%]"
                ],
                "Observed Empirical (%)": [f"{within_1s:.2f}%", f"{within_2s:.2f}%", f"{within_3s:.2f}%"],
                "Normal Rule Benchmark (%)": ["68.27%", "95.45%", "99.73%"],
                "Chebyshev Minimum Bound (%)": ["0.00% (k=1)", "≥ 75.00% (k=2)", "≥ 88.89% (k=3)"]
            })
            
            st.markdown(f"### 📊 Empirical Rule vs. Chebyshev's Inequality Evaluation ({selected_stock})")
            st.dataframe(bounds_table, use_container_width=True, hide_index=True)
            st.caption("Chebyshev's Theorem guarantees bounds for *any* arbitrary probability distribution. The empirical observations satisfy both Chebyshev's minimum requirements and closely mirror the Empirical 68-95-99.7 rule.")


# ==========================================
# 5. SAMPLING AND PROBABILITY (CLT)
# ==========================================
elif nav_selection == "5. Sampling & Probability (CLT)":
    st.markdown("## 🎲 Sampling Methods & Central Limit Theorem")
    st.markdown(f"Empirical validation of sampling methodologies and asymptotic convergence to normality using **{selected_stock}** return data.")
    
    if stock_return_series.empty:
        st.warning("Insufficient observations for sampling demonstrations.")
    else:
        population = stock_return_series.to_numpy()
        N = len(population)
        pop_mean = float(np.mean(population))
        pop_std = float(np.std(population, ddof=1))
        
        tab_s1, tab_s2 = st.tabs(["Sampling Methodologies Comparison", "Central Limit Theorem Simulation"])
        
        with tab_s1:
            st.markdown("### 🔍 Classical Probability Sampling Methods")
            
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                sample_size = st.slider("Sample Size (n)", min_value=10, max_value=min(150, N // 2), value=30, step=5)
            with col_ctrl2:
                seed_val = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
                
            # 1. Simple Random Sampling (SRS)
            srs_sample = stock_return_series.sample(n=sample_size, random_state=seed_val)
            
            # 2. Systematic Sampling (1-in-k)
            k_step = max(1, N // sample_size)
            systematic_sample = stock_return_series.iloc[::k_step].head(sample_size)
            
            # 3. Stratified Sampling by Calendar Year
            df_strat = stock_return_series.rename("Return").to_frame()
            df_strat["Year"] = df_strat.index.year
            n_years = df_strat["Year"].nunique()
            per_year_quota = max(1, int(np.ceil(sample_size / n_years)))
            stratified_sample = (
                df_strat.groupby("Year", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), per_year_quota), random_state=seed_val))
            )["Return"].head(sample_size)
            
            # Comparison Dataframe
            sampling_summary = pd.DataFrame([
                {
                    "Sampling Method": "Full Population (Ground Truth)",
                    "Observations (n)": N,
                    "Sample Mean (%)": pop_mean,
                    "Sample Std (%)": pop_std,
                    "Absolute Bias (|x̄ - μ|)": 0.000000
                },
                {
                    "Sampling Method": "Simple Random Sampling (SRS)",
                    "Observations (n)": len(srs_sample),
                    "Sample Mean (%)": float(srs_sample.mean()),
                    "Sample Std (%)": float(srs_sample.std(ddof=1)),
                    "Absolute Bias (|x̄ - μ|)": abs(float(srs_sample.mean()) - pop_mean)
                },
                {
                    "Sampling Method": f"Systematic Sampling (k={k_step})",
                    "Observations (n)": len(systematic_sample),
                    "Sample Mean (%)": float(systematic_sample.mean()),
                    "Sample Std (%)": float(systematic_sample.std(ddof=1)),
                    "Absolute Bias (|x̄ - μ|)": abs(float(systematic_sample.mean()) - pop_mean)
                },
                {
                    "Sampling Method": "Stratified Sampling (by Year)",
                    "Observations (n)": len(stratified_sample),
                    "Sample Mean (%)": float(stratified_sample.mean()),
                    "Sample Std (%)": float(stratified_sample.std(ddof=1)),
                    "Absolute Bias (|x̄ - μ|)": abs(float(stratified_sample.mean()) - pop_mean)
                }
            ])
            
            st.dataframe(sampling_summary.round(6), use_container_width=True, hide_index=True)
            
            # Visual comparison of samples
            fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
            ax.hist(population, bins=35, density=True, alpha=0.35, color="#64748b", label="Population Distribution")
            ax.hist(srs_sample, bins=15, density=True, alpha=0.5, color="#38bdf8", label=f"SRS Sample (n={len(srs_sample)})")
            ax.hist(stratified_sample, bins=15, density=True, alpha=0.5, color="#10b981", label=f"Stratified Sample (n={len(stratified_sample)})")
            ax.axvline(pop_mean, color="#ef4444", linestyle="--", linewidth=1.5, label="True Population Mean")
            ax.set_title("Comparison of Sample Densities vs. Population Ground Truth", fontsize=12, fontweight=600)
            ax.set_xlabel("Daily Return (%)", fontsize=10)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)
            
        with tab_s2:
            st.markdown("### 🔔 Central Limit Theorem (CLT) Simulation")
            st.markdown(r"""
            The **Central Limit Theorem** states that the sampling distribution of the sample mean ($\bar{X}$) approaches a 
            Normal distribution as $n$ increases, regardless of the underlying population's shape, with:
            $$\mathbb{E}[\bar{X}] = \mu, \quad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$
            """)
            
            col_clt1, col_clt2 = st.columns(2)
            with col_clt1:
                clt_n = st.select_slider("Sample Size per Draw (n)", options=[5, 15, 30, 50, 100], value=30)
            with col_clt2:
                clt_trials = st.select_slider("Number of Sample Trials (M)", options=[500, 1000, 2000], value=1000)
                
            # Perform Monte Carlo simulation
            np.random.seed(42)
            simulated_means = [
                np.mean(np.random.choice(population, size=clt_n, replace=True))
                for _ in range(clt_trials)
            ]
            simulated_means = np.array(simulated_means)
            
            mean_of_means = float(np.mean(simulated_means))
            empirical_se = float(np.std(simulated_means, ddof=1))
            theoretical_se = pop_std / np.sqrt(clt_n)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Population Mean (μ)", f"{pop_mean:.4f}%")
            with col_res2:
                st.metric("Mean of Sample Means (x̄̄)", f"{mean_of_means:.4f}%", delta=f"{mean_of_means - pop_mean:+.4f}%")
            with col_res3:
                st.metric("Theoretical vs. Empirical SE", f"{theoretical_se:.4f}%", delta=f"Obs: {empirical_se:.4f}%")
                
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
            count, bins, _ = ax.hist(simulated_means, bins=35, density=True, alpha=0.6, color="#0ea5e9", edgecolor="#0f172a", label=f"Simulated Distribution (M={clt_trials})")
            
            # Normal fit over sample means
            x_ax = np.linspace(simulated_means.min(), simulated_means.max(), 200)
            clt_norm = stats.norm.pdf(x_ax, loc=pop_mean, scale=theoretical_se)
            ax.plot(x_ax, clt_norm, color="#f59e0b", linestyle="--", linewidth=2.2, label=f"Theoretical Normal Curve: N(μ, σ/√n)")
            
            ax.axvline(pop_mean, color="#ef4444", linestyle="-", linewidth=1.5, label="Population Mean (μ)")
            ax.set_title(f"Sampling Distribution of Mean Returns (n={clt_n}, M={clt_trials} iterations)", fontsize=13, fontweight=600, pad=12)
            ax.set_xlabel("Sample Mean Return (%)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption(f"Theoretical Standard Error = {theoretical_se:.6f}% | Observed Empirical SE = {empirical_se:.6f}%. Notice that as n grows, standard error shrinks proportionally to 1/√n.")


# ==========================================
# 6. HYPOTHESIS TESTING
# ==========================================
elif nav_selection == "6. Hypothesis Testing":
    st.markdown("## 🧪 Inferential Hypothesis Testing & Confidence Intervals")
    st.markdown(f"Statistical significance testing on expected returns, benchmark comparisons, and return direction independence for **{selected_stock}**.")
    
    if stock_return_series.empty:
        st.warning("Insufficient data available for hypothesis testing.")
    else:
        tab_h1, tab_h2, tab_h3, tab_h4 = st.tabs([
            "One-Sample Tests & Z-Approximation",
            "Confidence Intervals",
            "Two-Sample & Paired T-Tests",
            "Chi-Square Tests (Goodness of Fit & Weekday)"
        ])
        
        # Calculations for Single Sample
        n_obs = len(stock_return_series)
        x_bar = float(stock_return_series.mean())
        s_std = float(stock_return_series.std(ddof=1))
        se_mean = s_std / np.sqrt(n_obs)
        
        with tab_h1:
            st.markdown("### 🎯 One-Sample Tests for Mean Daily Return")
            st.markdown(r"""
            **Null Hypothesis ($H_0$):** $\mu = 0.0\%$ (Mean daily return is zero)  
            **Alternative Hypothesis ($H_1$):** $\mu \neq 0.0\%$ (Two-sided test)
            """)
            
            alpha_level = st.selectbox("Significance Level (α)", options=[0.05, 0.01, 0.10], index=0)
            
            # One-sample t-test
            t_stat, t_pval = stats.ttest_1samp(stock_return_series, popmean=0.0)
            
            # Large-sample z approximation
            z_stat = (x_bar - 0.0) / se_mean
            z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                t_decision = "Reject H0 (Statistically Significant)" if t_pval < alpha_level else "Fail to Reject H0 (Insignificant)"
                st.markdown(f"""
                <div class="section-box">
                    <div class="section-title">One-Sample Student's t-Test</div>
                    • <strong>Degrees of Freedom (df):</strong> {n_obs - 1}<br>
                    • <strong>Test Statistic (t):</strong> {t_stat:.4f}<br>
                    • <strong>p-value:</strong> {t_pval:.6f}<br>
                    • <strong>Significance Level (α):</strong> {alpha_level}<br>
                    • <strong>Conclusion:</strong> <span class="badge {'badge-success' if t_pval < alpha_level else 'badge-warning'}">{t_decision}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with col_t2:
                z_decision = "Reject H0" if z_pval < alpha_level else "Fail to Reject H0"
                st.markdown(f"""
                <div class="section-box">
                    <div class="section-title">Large-Sample Z Approximation</div>
                    • <strong>Sample Size (n):</strong> {n_obs} (Justified since n ≫ 30)<br>
                    • <strong>Test Statistic (z):</strong> {z_stat:.4f}<br>
                    • <strong>Asymptotic p-value:</strong> {z_pval:.6f}<br>
                    • <strong>Conclusion:</strong> <span class="badge {'badge-success' if z_pval < alpha_level else 'badge-warning'}">{z_decision}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(r"""
            <div class="callout-info">
                <strong>Justification for Z Approximation:</strong> By the Central Limit Theorem, when the sample size is large ($n > 1000$), 
                the Student's $t$ distribution converges asymptotically to the standard normal $Z$ distribution. 
                However, caution is justified because the true population standard deviation $\sigma$ is unknown and estimated from the sample ($s$).
            </div>
            """, unsafe_allow_html=True)
            
        with tab_h2:
            st.markdown("### 📏 Confidence Intervals for Mean Return")
            st.markdown(r"Parametric two-sided confidence intervals for the population mean return $\mu$ across standard confidence levels.")
            
            ci_data = []
            for conf in [0.90, 0.95, 0.99]:
                a = 1 - conf
                t_crit = stats.t.ppf(1 - a / 2, df=n_obs - 1)
                moe = t_crit * se_mean
                ci_data.append({
                    "Confidence Level": f"{int(conf*100)}%",
                    "Critical Value (t*)": round(t_crit, 4),
                    "Standard Error (SE)": round(se_mean, 6),
                    "Margin of Error (%)": round(moe, 6),
                    "Lower Bound (%)": round(x_bar - moe, 6),
                    "Upper Bound (%)": round(x_bar + moe, 6),
                    "Contains Zero?": "Yes" if (x_bar - moe) <= 0 <= (x_bar + moe) else "No"
                })
                
            df_ci = pd.DataFrame(ci_data)
            st.dataframe(df_ci, use_container_width=True, hide_index=True)
            
            # Confidence interval error bar plot
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
            levels = [0.90, 0.95, 0.99]
            y_positions = [0, 1, 2]
            
            for idx, r in df_ci.iterrows():
                low = r["Lower Bound (%)"]
                high = r["Upper Bound (%)"]
                ax.plot([low, high], [idx, idx], marker="|", markersize=10, linewidth=2.5, label=f"{r['Confidence Level']} CI")
                
            ax.axvline(x_bar, color="#38bdf8", linestyle="-", label=f"Sample Mean ({x_bar:.4f}%)")
            ax.axvline(0, color="#ef4444", linestyle="--", label="Zero Return Benchmark")
            ax.set_yticks(y_positions)
            ax.set_yticklabels([r["Confidence Level"] for _, r in df_ci.iterrows()])
            ax.set_xlabel("Mean Daily Return (%)", fontsize=10)
            ax.set_title(f"Confidence Intervals for True Mean Daily Return: {selected_stock}", fontsize=12, fontweight=600)
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
            
        with tab_h3:
            st.markdown("### 👥 Two-Sample and Paired Tests")
            
            avail_pairs = [t for t in stock_tickers if t != selected_stock]
            if not avail_pairs:
                st.info("No alternate tickers available for pairwise comparison.")
            else:
                compare_target = st.selectbox("Compare with Stock:", options=avail_pairs, index=0)
                
                s1 = active_returns[selected_stock].dropna()
                s2 = active_returns[compare_target].dropna()
                
                # Independent Welch's Two-Sample t-test
                welch_res = stats.ttest_ind(s1, s2, equal_var=False, nan_policy="omit")
                
                # Paired t-test on matching dates
                paired_df = active_returns[[selected_stock, compare_target]].dropna()
                paired_res = stats.ttest_rel(paired_df[selected_stock], paired_df[compare_target])
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    w_dec = "Reject H0 (Means Differ)" if welch_res.pvalue < 0.05 else "Fail to Reject H0 (No Significant Difference)"
                    st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">Welch's Two-Sample Independent t-Test</div>
                        <strong>H0:</strong> Mean({selected_stock}) = Mean({compare_target})<br>
                        • <strong>{selected_stock} Mean:</strong> {s1.mean():.4f}%<br>
                        • <strong>{compare_target} Mean:</strong> {s2.mean():.4f}%<br>
                        • <strong>t-statistic:</strong> {welch_res.statistic:.4f}<br>
                        • <strong>p-value:</strong> {welch_res.pvalue:.6f}<br>
                        • <strong>Status (α=0.05):</strong> <span class="badge {'badge-success' if welch_res.pvalue < 0.05 else 'badge-warning'}">{w_dec}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_p2:
                    p_dec = "Reject H0 (Paired Diffs Non-zero)" if paired_res.pvalue < 0.05 else "Fail to Reject H0"
                    st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">Paired t-Test (Synchronous Dates)</div>
                        <strong>H0:</strong> Mean Daily Difference = 0<br>
                        • <strong>Synchronous Trading Days:</strong> {len(paired_df)}<br>
                        • <strong>Mean Difference:</strong> {(paired_df[selected_stock] - paired_df[compare_target]).mean():.4f}%<br>
                        • <strong>t-statistic:</strong> {paired_res.statistic:.4f}<br>
                        • <strong>p-value:</strong> {paired_res.pvalue:.6f}<br>
                        • <strong>Status (α=0.05):</strong> <span class="badge {'badge-success' if paired_res.pvalue < 0.05 else 'badge-warning'}">{p_dec}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with tab_h4:
            st.markdown("### 🎲 Chi-Square Tests of Categorical Market Return Behavior")
            
            # 1. Goodness of Fit (Direction Test: 50% Positive vs 50% Non-positive)
            pos_count = int((stock_return_series > 0).sum())
            non_pos_count = int((stock_return_series <= 0).sum())
            obs_counts = np.array([pos_count, non_pos_count])
            exp_counts = np.array([n_obs / 2, n_obs / 2])
            
            chi2_gof, p_gof = stats.chisquare(f_obs=obs_counts, f_exp=exp_counts)
            
            col_chi1, col_chi2 = st.columns(2)
            with col_chi1:
                st.markdown(f"""
                <div class="section-box">
                    <div class="section-title">Chi-Square Goodness-of-Fit Test</div>
                    <strong>H0:</strong> Daily returns have equal 50/50 probability of being Positive vs. Non-positive.<br><br>
                    • <strong>Positive Days:</strong> {pos_count} ({pos_count/n_obs*100:.1f}%)<br>
                    • <strong>Non-Positive Days:</strong> {non_pos_count} ({non_pos_count/n_obs*100:.1f}%)<br>
                    • <strong>Expected (50%):</strong> {exp_counts[0]:.1f}<br>
                    • <strong>Chi² Statistic:</strong> {chi2_gof:.4f}<br>
                    • <strong>p-value:</strong> {p_gof:.6f}<br>
                    • <strong>Conclusion (α=0.05):</strong> <span class="badge {'badge-success' if p_gof < 0.05 else 'badge-warning'}">{'Reject H0 (Directional Bias)' if p_gof < 0.05 else 'Fail to Reject H0 (Symmetric Chance)'}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # 2. Test of Independence (Day of Week Effect)
            with col_chi2:
                df_w = stock_return_series.rename("Return").to_frame()
                df_w["Weekday"] = df_w.index.day_name()
                df_w["Direction"] = np.where(df_w["Return"] > 0, "Positive", "Non-positive")
                
                # Order weekdays properly
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                contingency = pd.crosstab(df_w["Weekday"], df_w["Direction"])
                contingency = contingency.reindex([d for d in days_order if d in contingency.index])
                
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2_w, p_w, dof_w, exp_w = stats.chi2_contingency(contingency)
                    w_concl = "Reject H0 (Calendar Anomaly Found)" if p_w < 0.05 else "Fail to Reject H0 (No Day-of-Week Effect)"
                    st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">Chi-Square Independence Test (Weekday Effect)</div>
                        <strong>H0:</strong> Daily return direction is statistically independent of the day of the week.<br><br>
                        • <strong>Degrees of Freedom (df):</strong> {dof_w}<br>
                        • <strong>Chi² Statistic:</strong> {chi2_w:.4f}<br>
                        • <strong>p-value:</strong> {p_w:.6f}<br>
                        • <strong>Conclusion (α=0.05):</strong> <span class="badge {'badge-success' if p_w < 0.05 else 'badge-warning'}">{w_concl}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("#### Contingency Table: Day of Week vs. Return Direction")
            st.dataframe(contingency, use_container_width=True)


# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================
elif nav_selection == "7. Correlation Analysis":
    st.markdown("## 🔗 Cross-Asset Correlation & Covariance Matrix")
    st.markdown("Empirical co-movement, rank-order monotonicity, and beta-regression against market benchmark.")
    
    # Filter columns for correlation
    corr_cols = [c for c in stock_tickers if c in active_returns.columns]
    if use_benchmark and BENCHMARK in active_returns.columns:
        corr_cols.append(BENCHMARK)
        
    df_corr_data = active_returns[corr_cols].dropna()
    
    if df_corr_data.empty or len(corr_cols) < 2:
        st.warning("Insufficient multi-asset data available for correlation analysis.")
    else:
        tab_c1, tab_c2, tab_c3 = st.tabs(["Correlation & Covariance Matrices", "Stock vs. Benchmark Scatter (Beta)", "Pairwise Stock Scatter"])
        
        with tab_c1:
            matrix_choice = st.radio("Select Matrix View", ["Pearson Correlation (Linear)", "Spearman Correlation (Rank)", "Covariance Matrix"], horizontal=True)
            
            if matrix_choice == "Pearson Correlation (Linear)":
                mat = df_corr_data.corr(method="pearson")
                fmt = ".2f"
                title = "Pearson Correlation Matrix (Linear Co-movement)"
            elif matrix_choice == "Spearman Correlation (Rank)":
                mat = df_corr_data.corr(method="spearman")
                fmt = ".2f"
                title = "Spearman Correlation Matrix (Monotonic Rank)"
            else:
                mat = df_corr_data.cov()
                fmt = ".4f"
                title = "Covariance Matrix (Scale-dependent Co-variability)"
                
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
            sns.heatmap(mat, annot=True, cmap="coolwarm", center=0, fmt=fmt, ax=ax, cbar=True, linewidths=0.5, linecolor="#0f172a")
            ax.set_title(title, fontsize=12, fontweight=600, pad=12)
            st.pyplot(fig)
            plt.close(fig)
            
            st.dataframe(mat.round(4), use_container_width=True)
            
        with tab_c2:
            st.markdown(f"### 📈 {selected_stock} vs. Benchmark ({BENCHMARK}) Scatter & OLS Beta Regression")
            
            if BENCHMARK not in active_returns.columns:
                st.info("Benchmark data (^GSPC) is not present in the dataset.")
            else:
                paired_bm = active_returns[[selected_stock, BENCHMARK]].dropna()
                x_bm = paired_bm[BENCHMARK]
                y_stock = paired_bm[selected_stock]
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_bm, y_stock)
                spearman_r, spearman_p = stats.spearmanr(x_bm, y_stock)
                
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    st.metric("Beta (β - Systematic Risk)", f"{slope:.3f}", help="Slope of linear regression against S&P 500.")
                with col_b2:
                    st.metric("Alpha (α - Daily)", f"{intercept:+.4f}%", help="Intercept of daily return regression.")
                with col_b3:
                    st.metric("R² (Variance Explained)", f"{r_value**2:.3f}", help="Proportion of variance explained by market.")
                with col_b4:
                    st.metric("Pearson Correlation (r)", f"{r_value:.3f}", f"p={p_value:.1e}")
                    
                fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
                ax.scatter(x_bm, y_stock, color="#38bdf8", alpha=0.45, s=25, label=f"Daily Returns (N={len(paired_bm)})")
                
                # Regression line
                x_vals = np.linspace(x_bm.min(), x_bm.max(), 100)
                ax.plot(x_vals, intercept + slope * x_vals, color="#f59e0b", linewidth=2, label=f"OLS Fit: y = {slope:.2f}x + {intercept:.2f}% (R²={r_value**2:.2f})")
                
                ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
                ax.axvline(0, color="#475569", linestyle="--", linewidth=0.8)
                ax.set_title(f"{selected_stock} vs. S&P 500 Benchmark Return", fontsize=12, fontweight=600, pad=12)
                ax.set_xlabel("S&P 500 (^GSPC) Daily Return (%)", fontsize=10)
                ax.set_ylabel(f"{selected_stock} Daily Return (%)", fontsize=10)
                ax.legend(loc="upper left")
                st.pyplot(fig)
                plt.close(fig)
                
        with tab_c3:
            st.markdown("### 🔀 Pairwise Stock-to-Stock Scatter Analysis")
            avail_pairs = [t for t in stock_tickers if t != selected_stock]
            if not avail_pairs:
                st.info("No alternate tickers available.")
            else:
                pair_asset = st.selectbox("Select Asset to Compare:", options=avail_pairs, index=0)
                pair_df = active_returns[[selected_stock, pair_asset]].dropna()
                
                pr_r, pr_p = stats.pearsonr(pair_df[selected_stock], pair_df[pair_asset])
                sp_r, sp_p = stats.spearmanr(pair_df[selected_stock], pair_df[pair_asset])
                
                col_sc1, col_sc2 = st.columns(2)
                with col_sc1:
                    st.metric("Pearson Correlation", f"{pr_r:.4f}", f"p-value: {pr_p:.1e}")
                with col_sc2:
                    st.metric("Spearman Rank Correlation", f"{sp_r:.4f}", f"p-value: {sp_p:.1e}")
                    
                fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
                ax.scatter(pair_df[pair_asset], pair_df[selected_stock], color="#a855f7", alpha=0.45, s=25)
                ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
                ax.axvline(0, color="#475569", linestyle="--", linewidth=0.8)
                ax.set_xlabel(f"{pair_asset} Daily Return (%)", fontsize=10)
                ax.set_ylabel(f"{selected_stock} Daily Return (%)", fontsize=10)
                ax.set_title(f"Pairwise Dispersion: {selected_stock} vs. {pair_asset}", fontsize=12, fontweight=600)
                st.pyplot(fig)
                plt.close(fig)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================
elif nav_selection == "8. Model Evaluation":
    st.markdown("## 🏆 Empirical Model Evaluation & Benchmarks")
    st.markdown("Actual saved performance metrics across fitted regression algorithms and classification pipelines evaluated on the chronologically split holdout test set.")
    
    df_reg_comp = data_dict.get("regression_comparison")
    df_clf_comp = data_dict.get("classification_comparison")
    df_cm = data_dict.get("confusion_matrix")
    
    if df_reg_comp is None or df_clf_comp is None:
        st.warning("Model comparison datasets not found in stock_analyzer_data/.")
    else:
        tab_m1, tab_m2, tab_m3 = st.tabs(["Regression Model Comparison", "Classification Model Comparison", "Test Confusion Matrix & Report"])
        
        with tab_m1:
            st.markdown("### 📉 Regression Models (Target: Next-Day Daily Return %)")
            st.markdown("Sorted in ascending order of **Root Mean Squared Error (RMSE)**:")
            
            # Highlight best model
            st.dataframe(df_reg_comp.style.highlight_min(subset=["RMSE", "MAE"], color="#065f46").highlight_max(subset=["R2 Score"], color="#065f46"), use_container_width=True)
            
            # Bar Chart Comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
            
            # RMSE & MAE
            models = df_reg_comp["Model"]
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, df_reg_comp["RMSE"], width, label="RMSE", color="#38bdf8")
            ax1.bar(x + width/2, df_reg_comp["MAE"], width, label="MAE", color="#818cf8")
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
            ax1.set_ylabel("Error Metric (%)", fontsize=10)
            ax1.set_title("Regression Error Metrics (Lower is Better)", fontsize=11, fontweight=600)
            ax1.legend()
            
            # R2 Score
            colors_r2 = ['#34d399' if v >= 0 else '#f87171' for v in df_reg_comp["R2 Score"]]
            ax2.bar(models, df_reg_comp["R2 Score"], color=colors_r2, width=0.5)
            ax2.axhline(0, color="#64748b", linestyle="--", linewidth=0.8)
            ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
            ax2.set_ylabel("R² Score", fontsize=10)
            ax2.set_title("Coefficient of Determination (R²)", fontsize=11, fontweight=600)
            
            st.pyplot(fig)
            plt.close(fig)
            
            best_reg_model = df_reg_comp.iloc[0]["Model"]
            st.success(f"**Best Performing Regression Model:** `{best_reg_model}` with lowest RMSE of **{df_reg_comp.iloc[0]['RMSE']:.4f}%**.")
            st.caption("Notice that financial market returns have low signal-to-noise ratios; R² scores near zero are characteristic of efficient market dynamics where historical price patterns do not easily predict immediate next-day linear returns.")
            
        with tab_m2:
            st.markdown("### 🎯 Classification Models (Target: Direction > 0)")
            st.markdown("Evaluated on holdout test set with balanced class weighting. Sorted by **F1-Score**:")
            
            st.dataframe(df_clf_comp.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#065f46"), use_container_width=True)
            
            # Classification Metrics Bar Chart
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
            models_clf = df_clf_comp["Model"]
            x = np.arange(len(models_clf))
            width = 0.2
            
            ax.bar(x - 1.5*width, df_clf_comp["Accuracy"], width, label="Accuracy", color="#38bdf8")
            ax.bar(x - 0.5*width, df_clf_comp["Precision"], width, label="Precision", color="#34d399")
            ax.bar(x + 0.5*width, df_clf_comp["Recall"], width, label="Recall", color="#f59e0b")
            ax.bar(x + 1.5*width, df_clf_comp["F1-Score"], width, label="F1-Score", color="#a855f7")
            
            ax.set_xticks(x)
            ax.set_xticklabels(models_clf, fontsize=10)
            ax.set_ylabel("Score (0 to 1.0)", fontsize=10)
            ax.set_title("Classification Performance Metrics Comparison", fontsize=12, fontweight=600)
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
            
            best_clf_model = df_clf_comp.iloc[0]["Model"]
            st.success(f"**Best Performing Classification Model:** `{best_clf_model}` with top F1-Score of **{df_clf_comp.iloc[0]['F1-Score']:.4f}** and Accuracy of **{df_clf_comp.iloc[0]['Accuracy']*100:.2f}%**.")
            
        with tab_m3:
            st.markdown("### 🔲 Holdout Test Confusion Matrix")
            if df_cm is not None:
                col_cm1, col_cm2 = st.columns([1, 1.2])
                with col_cm1:
                    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
                    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"})
                    ax.set_title("Holdout Test Confusion Matrix", fontsize=11, fontweight=600, pad=10)
                    ax.set_xlabel("Predicted Class", fontsize=10)
                    ax.set_ylabel("Actual Class", fontsize=10)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with col_cm2:
                    tn = df_cm.iloc[0, 0]
                    fp = df_cm.iloc[0, 1]
                    fn = df_cm.iloc[1, 0]
                    tp = df_cm.iloc[1, 1]
                    tot = tn + fp + fn + tp
                    
                    st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">Diagnostic Metrics Breakdown</div>
                        • <strong>Holdout Test Sample Size:</strong> {tot} observations (chronological 20% split)<br>
                        • <strong>True Negatives (TN):</strong> {tn} (Correctly classified non-positive days)<br>
                        • <strong>False Positives (FP):</strong> {fp} (Type I Error)<br>
                        • <strong>False Negatives (FN):</strong> {fn} (Type II Error)<br>
                        • <strong>True Positives (TP):</strong> {tp} (Correctly classified positive days)<br>
                        • <strong>Sensitivity / Recall:</strong> {tp / (tp + fn) * 100:.2f}%<br>
                        • <strong>Specificity:</strong> {tn / (tn + fp) * 100:.2f}%<br>
                        • <strong>Precision:</strong> {tp / (tp + fp) * 100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Saved confusion matrix data is not available.")


# ==========================================
# 9. PREDICTIONS
# ==========================================
elif nav_selection == "9. Predictive Inference":
    st.markdown("## 🔮 Next-Day Predictive Inference Engine")
    st.markdown("Execution of trained Scikit-Learn pipelines (`Lasso Regression` and `Logistic Regression`) utilizing strictly verified feature vectors.")
    
    reg_pipeline = model_dict.get("regression")
    clf_pipeline = model_dict.get("classification")
    
    if reg_pipeline is None or clf_pipeline is None:
        st.error("Model pipelines could not be loaded. Please ensure `.pkl` model files exist in `models/` or `scratch/`.")
        if model_errors:
            for e in model_errors:
                st.write(f"- {e}")
    else:
        # Expected features in strict order
        EXPECTED_FEATURES = [
            'Return_Lag1', 'Return_Lag2', 'Return_Lag5',
            'Volatility_5D', 'Volatility_20D',
            'Moving_Average_5D', 'Moving_Average_20D',
            'Benchmark_Return'
        ]
        
        st.markdown("""
        <div class="callout-info">
            <strong>Fitted Pipeline Architecture:</strong><br>
            • <strong>Regression:</strong> SimpleImputer (median) → StandardScaler → Lasso(α=0.1)<br>
            • <strong>Classification:</strong> SimpleImputer (median) → StandardScaler → LogisticRegression (balanced weights)<br>
            • <strong>Input Feature Vector (8 strictly verified dimensions):</strong> <code>Return_Lag1, Return_Lag2, Return_Lag5, Volatility_5D, Volatility_20D, Moving_Average_5D, Moving_Average_20D, Benchmark_Return</code>.
        </div>
        """, unsafe_allow_html=True)
        
        pred_mode = st.radio(
            "Select Inference Mode:",
            ["A. Latest Historical Market Features (Real Data)", "B. Interactive Scenario Simulator (Stress Test)"],
            horizontal=True
        )
        
        # Prepare feature vector
        feature_df = None
        obs_date_str = "Custom Scenario"
        
        if pred_mode.startswith("A"):
            if df_model_data is not None and not df_model_data.empty:
                # Latest row from modeling dataset
                latest_row = df_model_data[EXPECTED_FEATURES].iloc[[-1]]
                obs_date_str = df_model_data.index[-1].strftime('%Y-%m-%d') if isinstance(df_model_data.index[-1], pd.Timestamp) else str(df_model_data.index[-1])
                feature_df = latest_row.copy()
            else:
                # Construct from daily returns for selected stock
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
                    feature_df = temp[EXPECTED_FEATURES].iloc[[-1]].copy()
                    obs_date_str = temp.index[-1].strftime('%Y-%m-%d')
                else:
                    st.error("Unable to compute feature vector from returns data.")
                    
            if feature_df is not None:
                st.markdown(f"### 📅 Market Feature Vector as of `{obs_date_str}` (Asset: **{selected_stock}**)")
                st.dataframe(feature_df.round(4), use_container_width=True)
                
        else:
            st.markdown("### 🎛️ Interactive Market Condition Controls")
            st.caption("Adjust hypothetical lagged returns and rolling volatilities to test model reactions:")
            
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                f_lag1 = st.slider("Return Lag 1 Day (%)", -10.0, 10.0, 0.5, 0.1)
                f_lag2 = st.slider("Return Lag 2 Days (%)", -10.0, 10.0, -0.2, 0.1)
            with c_f2:
                f_lag5 = st.slider("Return Lag 5 Days (%)", -10.0, 10.0, 1.0, 0.1)
                f_bm = st.slider("Benchmark Return (%)", -5.0, 5.0, 0.2, 0.1)
            with c_f3:
                f_vol5 = st.slider("Volatility 5-Day (σ %)", 0.1, 8.0, 1.5, 0.1)
                f_vol20 = st.slider("Volatility 20-Day (σ %)", 0.1, 8.0, 1.8, 0.1)
            with c_f4:
                f_ma5 = st.slider("Moving Avg 5-Day (%)", -5.0, 5.0, 0.3, 0.1)
                f_ma20 = st.slider("Moving Avg 20-Day (%)", -5.0, 5.0, 0.1, 0.1)
                
            feature_df = pd.DataFrame([{
                'Return_Lag1': f_lag1,
                'Return_Lag2': f_lag2,
                'Return_Lag5': f_lag5,
                'Volatility_5D': f_vol5,
                'Volatility_20D': f_vol20,
                'Moving_Average_5D': f_ma5,
                'Moving_Average_20D': f_ma20,
                'Benchmark_Return': f_bm
            }])[EXPECTED_FEATURES]
            
        # Execute Predictions
        if feature_df is not None and not feature_df.empty:
            try:
                pred_reg = reg_pipeline.predict(feature_df)[0]
                pred_clf = clf_pipeline.predict(feature_df)[0]
                pred_proba = clf_pipeline.predict_proba(feature_df)[0] if hasattr(clf_pipeline, "predict_proba") else None
                
                st.markdown("---")
                st.markdown("### 🎯 Model Pipeline Forecast Outputs")
                
                col_out1, col_out2, col_out3 = st.columns(3)
                
                with col_out1:
                    reg_sign = "+" if pred_reg >= 0 else ""
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Next-Day Return</div>
                        <div class="metric-value" style="color: {'#34d399' if pred_reg >= 0 else '#f87171'}">{reg_sign}{pred_reg:.4f}%</div>
                        <div class="metric-sub neutral">Fitted Lasso Regression Pipeline</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_out2:
                    dir_label = "Positive (Up)" if pred_clf == 1 else "Non-positive (Flat / Down)"
                    dir_color = "#34d399" if pred_clf == 1 else "#f87171"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Market Direction</div>
                        <div class="metric-value" style="color: {dir_color}; font-size: 1.35rem;">{dir_label}</div>
                        <div class="metric-sub neutral">Fitted Logistic Classifier Pipeline</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_out3:
                    if pred_proba is not None:
                        prob_up = pred_proba[1] * 100
                        prob_down = pred_proba[0] * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Direction Confidence</div>
                            <div class="metric-value">{prob_up:.1f}% Up</div>
                            <div class="metric-sub neutral">Down: {prob_down:.1f}% | Up: {prob_up:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Direction Confidence</div>
                            <div class="metric-value">N/A</div>
                            <div class="metric-sub neutral">Probabilities unavailable</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                
        # Prominent Academic Disclaimer
        st.markdown("""
        <div class="callout-disclaimer">
            <strong>⚠️ IMPORTANT ACADEMIC & FINANCIAL DISCLAIMER:</strong><br>
            These predictions are generated by statistical regression and classification models trained on historical equity return data for academic and demonstration purposes only. Financial markets are governed by complex macroeconomic variables, regime shifts, and stochastic noise. <strong>Under no circumstances should these model outputs be interpreted as financial advice, investment recommendations, or reliable trading signals.</strong>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
    <strong>Stock Market Return Statistical Analyzer</strong> | College Project Edition<br>
    Built with Streamlit, Pandas, NumPy, Matplotlib, Seaborn, SciPy, and Scikit-Learn.
</div>
""", unsafe_allow_html=True)

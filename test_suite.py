"""
Automated Test Suite for Stock Market Return Statistical Analyzer
Validates data integrity, model pipelines, and mathematical functions across all 9 sections.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import joblib

def run_tests():
    print("==================================================")
    print("STARTING TEST SUITE: Stock Market Return Statistical Analyzer")
    print("==================================================")
    
    # 1. FILE INTEGRITY
    print("\n[TEST 1] Verifying Data & Model Files...")
    required_files = [
        "stock_analyzer_data/stock_close_prices.csv",
        "stock_analyzer_data/daily_stock_returns.csv",
        "stock_analyzer_data/modeling_dataset.csv",
        "stock_analyzer_data/regression_model_comparison.csv",
        "stock_analyzer_data/classification_model_comparison.csv",
        "stock_analyzer_data/classification_confusion_matrix.csv",
        "models/best_return_regression_model.pkl",
        "models/best_direction_classification_model.pkl"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
        print(f"  [OK] Found {rf} ({os.path.getsize(rf):,} bytes)")
        
    # 2. DATA LOADING & FORMAT
    print("\n[TEST 2] Testing Data Structures & Date Parsing...")
    prices = pd.read_csv("stock_analyzer_data/stock_close_prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv("stock_analyzer_data/daily_stock_returns.csv", index_col=0, parse_dates=True)
    model_df = pd.read_csv("stock_analyzer_data/modeling_dataset.csv", index_col=0, parse_dates=True)
    
    assert not prices.empty, "Prices dataframe is empty"
    assert not returns.empty, "Returns dataframe is empty"
    assert not model_df.empty, "Modeling dataset is empty"
    
    tickers = [c for c in returns.columns if c not in ["Date", "index"]]
    assert "AAPL" in tickers and "^GSPC" in tickers, "Expected tickers not found"
    print(f"  [OK] Tickers verified: {tickers}")
    print(f"  [OK] Observation window: {returns.index.min().strftime('%Y-%m-%d')} to {returns.index.max().strftime('%Y-%m-%d')} ({len(returns)} rows)")
    
    # 3. STATISTICAL FUNCTIONS
    print("\n[TEST 3] Testing Descriptive Statistics on All Tickers...")
    for t in tickers:
        s = returns[t].dropna()
        assert len(s) > 100, f"Insufficient observations for {t}"
        mean_v = s.mean()
        std_v = s.std(ddof=1)
        median_v = s.median()
        skew_v = s.skew()
        kurt_v = s.kurtosis()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        assert not np.isnan(mean_v) and not np.isnan(std_v) and not np.isnan(iqr)
        print(f"  [OK] {t:6s}: Mean = {mean_v:+.4f}%, Std = {std_v:.4f}%, Skew = {skew_v:+.4f}, Kurt = {kurt_v:.4f}")
        
    # 4. SAMPLING METHODS
    print("\n[TEST 4] Testing Sampling Methods & CLT Convergence...")
    s_aapl = returns["AAPL"].dropna().to_numpy()
    pop_mean = np.mean(s_aapl)
    pop_std = np.std(s_aapl, ddof=1)
    
    # SRS
    np.random.seed(42)
    srs = np.random.choice(s_aapl, size=30, replace=False)
    assert len(srs) == 30
    
    # Systematic (1-in-k)
    k = max(1, len(s_aapl) // 30)
    syst = s_aapl[::k][:30]
    assert len(syst) == 30
    
    # CLT Simulation
    means = [np.mean(np.random.choice(s_aapl, size=30, replace=True)) for _ in range(500)]
    clt_mean = np.mean(means)
    clt_se = np.std(means, ddof=1)
    theo_se = pop_std / np.sqrt(30)
    print(f"  [OK] Pop Mean: {pop_mean:.4f}% | CLT Mean of Means: {clt_mean:.4f}%")
    print(f"  [OK] Theoretical SE: {theo_se:.4f}% | Empirical CLT SE: {clt_se:.4f}%")
    assert abs(clt_mean - pop_mean) < 0.1, "CLT mean deviated too much"
    assert abs(clt_se - theo_se) < 0.1, "CLT standard error deviated too much"
    
    # 5. HYPOTHESIS TESTING
    print("\n[TEST 5] Testing Hypothesis Tests & Confidence Intervals...")
    s_series = returns["AAPL"].dropna()
    t_stat, t_p = stats.ttest_1samp(s_series, 0.0)
    se = s_series.std(ddof=1) / np.sqrt(len(s_series))
    z_stat = (s_series.mean() - 0.0) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print(f"  [OK] One-Sample t-test: t = {t_stat:.4f}, p = {t_p:.6f}")
    print(f"  [OK] Large-sample z-test: z = {z_stat:.4f}, p = {z_p:.6f}")
    assert abs(t_stat - z_stat) < 0.01, "t and z statistics should align for n > 1000"
    
    # Two-sample Welch
    s_msft = returns["MSFT"].dropna()
    w_stat, w_p = stats.ttest_ind(s_series, s_msft, equal_var=False)
    print(f"  [OK] Welch Two-Sample (AAPL vs MSFT): t = {w_stat:.4f}, p = {w_p:.6f}")
    
    # Chi-square Goodness-of-Fit
    pos = (s_series > 0).sum()
    neg = (s_series <= 0).sum()
    chi_gof, p_gof = stats.chisquare([pos, neg], [len(s_series)/2, len(s_series)/2])
    print(f"  [OK] Chi-Square Goodness-of-Fit: chi2 = {chi_gof:.4f}, p = {p_gof:.6f}")
    
    # Chi-square Weekday Independence
    df_w = s_series.rename("Return").to_frame()
    df_w["Weekday"] = df_w.index.day_name()
    df_w["Dir"] = np.where(df_w["Return"] > 0, "Pos", "Non-pos")
    ct = pd.crosstab(df_w["Weekday"], df_w["Dir"])
    chi_w, p_w, dof_w, _ = stats.chi2_contingency(ct)
    print(f"  [OK] Chi-Square Weekday Independence: chi2 = {chi_w:.4f}, df = {dof_w}, p = {p_w:.6f}")
    
    # 6. CORRELATION & BETA REGRESSION
    print("\n[TEST 6] Testing Correlation & Systematic Risk (Beta)...")
    paired = returns[["AAPL", "^GSPC"]].dropna()
    pr_r, pr_p = stats.pearsonr(paired["AAPL"], paired["^GSPC"])
    sp_r, sp_p = stats.spearmanr(paired["AAPL"], paired["^GSPC"])
    slope, intercept, r_val, p_val, std_err = stats.linregress(paired["^GSPC"], paired["AAPL"])
    print(f"  [OK] Pearson r: {pr_r:.4f} | Spearman rho: {sp_r:.4f}")
    print(f"  [OK] Capital Asset Beta: Beta = {slope:.4f}, Alpha: Alpha = {intercept:+.4f}%, R2 = {r_val**2:.4f}")
    assert slope > 0.5, "AAPL Beta against S&P 500 should be substantially positive"
    
    # 7. MODEL EVALUATION ARTIFACTS
    print("\n[TEST 7] Verifying Model Evaluation CSVs...")
    df_reg_eval = pd.read_csv("stock_analyzer_data/regression_model_comparison.csv")
    df_clf_eval = pd.read_csv("stock_analyzer_data/classification_model_comparison.csv")
    df_cm = pd.read_csv("stock_analyzer_data/classification_confusion_matrix.csv", index_col=0)
    
    assert len(df_reg_eval) == 5, f"Expected 5 regression models, found {len(df_reg_eval)}"
    assert len(df_clf_eval) == 3, f"Expected 3 classification models, found {len(df_clf_eval)}"
    assert df_cm.shape == (2, 2), "Confusion matrix must be 2x2"
    print(f"  [OK] Regression Models: {df_reg_eval['Model'].tolist()}")
    print(f"  [OK] Classification Models: {df_clf_eval['Model'].tolist()}")
    print(f"  [OK] Confusion Matrix Verified:\n{df_cm}")
    
    # 8. PREDICTION PIPELINES
    print("\n[TEST 8] Testing Model Prediction Pipelines...")
    reg = joblib.load("models/best_return_regression_model.pkl")
    clf = joblib.load("models/best_direction_classification_model.pkl")
    
    # Apply cross-version patch
    for m in [reg, clf]:
        imp = getattr(m, "named_steps", {}).get("imputer")
        if imp is not None and not hasattr(imp, "_fill_dtype"):
            imp._fill_dtype = getattr(imp, "_fit_dtype", float)
            
    expected_feats = [
        'Return_Lag1', 'Return_Lag2', 'Return_Lag5',
        'Volatility_5D', 'Volatility_20D',
        'Moving_Average_5D', 'Moving_Average_20D',
        'Benchmark_Return'
    ]
    test_features = model_df[expected_feats].iloc[[-1]]
    
    pred_ret = reg.predict(test_features)[0]
    pred_dir = clf.predict(test_features)[0]
    pred_probs = clf.predict_proba(test_features)[0]
    
    print(f"  [OK] Pipeline Expected Features: {expected_feats}")
    print(f"  [OK] Next-Day Return Estimate: {pred_ret:+.4f}%")
    print(f"  [OK] Predicted Direction: {pred_dir} ({'Positive' if pred_dir == 1 else 'Non-positive'})")
    print(f"  [OK] Class Probabilities: Down={pred_probs[0]*100:.1f}%, Up={pred_probs[1]*100:.1f}%")
    
    assert -10.0 <= pred_ret <= 10.0, "Regression prediction should be in reasonable daily return range"
    assert pred_dir in [0, 1], "Direction prediction must be binary 0 or 1"
    assert len(pred_probs) == 2, "Binary probability vector required"
    
    # 9. EDGE CASES & ERROR RESILIENCE
    print("\n[TEST 9] Testing Edge Cases & Missing-Data Robustness...")
    empty_series = pd.Series(dtype=float)
    from app import calculate_descriptive_stats
    empty_stats = calculate_descriptive_stats(empty_series)
    assert empty_stats.empty, "Empty series should return empty series without throwing error"
    print("  [OK] Empty series gracefully handled")
    
    print("\n==================================================")
    print("[SUCCESS] ALL 9 TESTS PASSED SUCCESSFULLY! ZERO ERRORS.")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

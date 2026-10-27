# Stock Market Return Statistical Analyzer

An interactive, modern light-mode financial analytics and statistical dashboard built with **Streamlit**, **Plotly**, **Pandas**, **SciPy**, and **Scikit-Learn** for university quantitative market research and academic presentations.

---

## 📌 Project Overview

This dashboard analyzes 5-year daily adjusted closing price and percentage return series for mega-cap technology equities (**AAPL**, **MSFT**, **NVDA**, **AMZN**, **GOOGL**) alongside the **S&P 500 benchmark (`^GSPC`)**. It implements classical descriptive statistics, probability distributions, sampling theory (Central Limit Theorem), inferential hypothesis tests, correlation structures, and machine learning predictive pipelines in a clean, publication-ready light interface.

---

## 🚀 Key Dashboard Modules

1. **Overview:** Executive KPIs (Closing Price, Cumulative Return, Daily Mean, Daily Volatility), normalized performance chart (Base = 100) vs. S&P 500, dynamic factual observations, and multi-asset risk snapshot.
2. **Historical Trends:** Price action with 20-day and 50-day moving averages, growth of an initial $10,000 investment, and daily return fluctuations with rolling peak-to-trough drawdowns.
3. **Statistics:** Classical statistical moments (mean, median, mode, variance, standard deviation, quartiles, IQR, skewness, and kurtosis) with cross-asset comparative tables.
4. **Return Distribution:** Empirical return histogram with Kernel Density Estimation (KDE), theoretical Gaussian normal curve, outlier boxplots, and Empirical Rule vs. Chebyshev bound evaluation.
5. **Sampling & CLT:** Live probability sampling demonstrations (Simple Random, Systematic, Stratified by Year) and an interactive Central Limit Theorem Monte Carlo simulation.
6. **Hypothesis Testing:** One-sample $t$-test ($H_0: \mu = 0$), large-sample $z$ approximation, confidence intervals (90%, 95%, 99%), Welch's two-sample $t$-test, paired $t$-test on matching trading dates, and Chi-Square tests (Direction Goodness-of-Fit & Weekday Independence).
7. **Correlation Analysis:** Pearson linear correlation, Spearman monotonic rank correlation, covariance matrix, interactive heatmaps, and stock vs. benchmark scatter regression with beta ($\beta$) and alpha ($\alpha$).
8. **Model Performance:** Real performance metrics loaded from historical evaluation datasets across 5 regression models (MAE, RMSE, $R^2$) and 3 classification models (Accuracy, Precision, Recall, F1-Score), including the holdout test-set confusion matrix.
9. **Predictions:** Next-day return forecasting and market direction classification powered by trained Scikit-Learn pipelines (`Lasso Regression` and `Logistic Regression`) using 8 exact technical features. Includes latest historical market data and custom scenario simulation.

---

## 📁 Repository Structure

```text
├── app.py                                            # Main Streamlit Dashboard Application
├── requirements.txt                                  # Python dependencies
├── README.md                                         # Project documentation
├── Stock_Market_Return_Statistical_Analyzer.ipynb    # Research & analysis Jupyter Notebook
├── models/
│   ├── best_return_regression_model.pkl              # Fitted Lasso regression pipeline
│   └── best_direction_classification_model.pkl       # Fitted Logistic regression pipeline
└── stock_analyzer_data/
    ├── stock_close_prices.csv                        # Daily adjusted closing prices (5Y)
    ├── daily_stock_returns.csv                       # Percentage daily returns (5Y)
    ├── modeling_dataset.csv                          # Feature-engineered predictive dataset
    ├── regression_model_comparison.csv               # Model performance metrics (Regression)
    ├── classification_model_comparison.csv           # Model performance metrics (Classification)
    ├── classification_confusion_matrix.csv           # Holdout test confusion matrix
    └── sp500_ticker_list.csv                         # S&P 500 company ticker reference
```

---

## 🛠️ Installation & Setup

### 1. Clone or Open Workspace
Ensure you are in the project root directory containing `app.py`.

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit web server:
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if your Python Scripts folder is added to your system `PATH`.)*

Once executed, the application will automatically open in your default browser at `http://localhost:8501`.

---

## 🔬 Statistical & Machine Learning Notes

- **Features Used in Models (8 Exact Features):**
  - `Return_Lag1`, `Return_Lag2`, `Return_Lag5`: Lagged returns capturing short-term momentum / mean reversion.
  - `Volatility_5D`, `Volatility_20D`: Rolling standard deviations capturing volatility clustering.
  - `Moving_Average_5D`, `Moving_Average_20D`: Rolling return averages.
  - `Benchmark_Return`: S&P 500 contemporaneous/lagged market return.
- **Data Scaling & Imputation:** Model pipelines integrate median imputation and standard scaling inside fitted `Pipeline` objects to avoid data leakage.
- **Large-Sample Z Test Justification:** By the Central Limit Theorem, for $n > 1000$ the distribution of the sample mean is asymptotically normal. While the population variance $\sigma^2$ is replaced by sample variance $s^2$, the $z$ test provides a benchmark comparison against Student's $t$.

---

## ⚠️ Academic Disclaimer

This software application is built strictly for educational, college presentation, and statistical demonstration purposes. Stock market forecasting is subject to high stochasticity, regime shifts, and non-stationarity. None of the model outputs, forecasts, or statistical metrics constitute financial, investment, or trading advice.

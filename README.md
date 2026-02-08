# SPY Next-Day Direction Prediction

**A reproducible Quant Research ML pipeline with time-aware validation, walk-forward evaluation, and signal-to-strategy backtesting**

---

## 0. TL;DR

This repository implements an end-to-end quant research pipeline for predicting **SPY’s next-day close-to-close direction** using only **historical daily OHLCV** data. The focus is not “chasing a high accuracy number,” but building a **leakage-safe**, **time-consistent**, and **fully reproducible** research workflow: data → labels → features → time-aware validation → walk-forward evaluation → trading diagnostics (deciles, thresholds, costs). The results highlight the **structural limits of traditional ML baselines** in short-horizon market direction prediction and motivate a rigorous transition to **deep sequence models** under the same evaluation framework.

## Key Results (Walk-Forward Summary)

Yearly expanding-window walk-forward (mean ± std across years):
- Accuracy: **0.5058 ± 0.0367**
- AUC: **0.5061 ± 0.0391**
- LogLoss: **0.7068 ± 0.0145**
- Baseline Accuracy: **0.5515 ± 0.0407**

Interpretation:
- Under leakage-safe, time-consistent evaluation, next-day direction signal is **weak and unstable across years** (AUC ≈ 0.5).
- The naive baseline can outperform accuracy, reinforcing that **headline accuracy is not a reliable indicator** for this task.
- This motivates focusing on ranking diagnostics (deciles), regime robustness, and exploring sequence models (LSTM/Transformer) under the same walk-forward framework.


---

## 1. Motivation (Self-driven research framing)

Short-horizon market direction prediction is a canonical but notoriously noisy problem. I built this project independently from scratch to demonstrate:

* **Quant research hygiene**: correct target construction, strict time alignment, and leakage prevention
* **Engineering discipline**: modular scripts, consistent data schema, reproducible outputs
* **Model evaluation maturity**: time-aware splits, walk-forward testing, and trading-oriented diagnostics
* **Research thinking**: interpreting negative/weak results and extracting actionable next steps

Rather than presenting “pretty metrics,” this repo emphasizes **why the baseline struggles**, **when it fails**, and **what to do next**.

---

## 2. Repository layout & reproducibility

A typical layout:

* `src/` : research scripts (data → features → validation → backtests)
* `data/` : raw/intermediate CSV artifacts
* `results/` : metrics tables, predictions, and plots
* `notebooks/` : optional exploratory analysis

**Repro principle:** every stage writes an explicit artifact (CSV/plot) so the pipeline is auditable and rerunnable.

---

## 3. Problem formulation (what exactly is predicted)

### 3.1 Task

Binary classification: predict whether SPY closes higher tomorrow than today.

### 3.2 Strict information set

* Inputs: **daily OHLCV** only (historically available at time (t))
* No random shuffling; training always precedes testing in time
* All features are computed using data up to (t) (no look-ahead)

---

## 4. Data collection (`data_collection.py`)

### What it does

* Downloads **SPY** daily data from **yfinance** starting at **2000-01-01**
* Columns: `Date, Open, High, Low, Close, Volume`
* Sorts ascending by date
* Saves: `data/raw_spy.csv`

### Practical checks

* Confirm date range and row count
* Confirm OHLCV fields are complete and correctly typed

---

## 5. Data cleaning & schema standardization (`data_cleaning.py`)

### What it does

* Reads the raw data (e.g., `raw_spy.csv` / `sp500_data.csv`)
* Detects and standardizes the date column, converts to datetime index
* Keeps canonical OHLCV schema: `Open/High/Low/Close/Volume`
* Numeric casting, remove missing/duplicates, sort by date
* Saves: `data/sp500_data_clean.csv`
* Prints: before/after row counts, date ranges, head/tail samples (sanity checks)

### Typical failure modes I handled

* **Column name mismatch** (date not named `Date`, close not named `Close`)
  → solve by enforcing a consistent schema at the cleaning stage
* **Rolling-induced NaNs later in the pipeline**
  → accept as normal and handle with deterministic truncation after feature generation

---

## 6. Target labeling (`make_target.py`) — preventing look-ahead bias

The most dangerous bug in financial ML is subtle label/feature misalignment. This step is designed to be auditable.

### 6.1 Definitions

For each trading day (t):
$$
Target_t = \mathbb{1}(Close_{t+1} > Close_t)
$$
$$
Return_{1d,t} = \frac{Close_{t+1}}{Close_t} - 1
$$

Implementation: `shift(-1)` on `Close` to obtain (Close_{t+1}), then compute `Target` and `Return_1d`.

### 6.2 Why keep `Return_1d`

Accuracy alone can hide whether the model differentiates **small moves** vs **large moves**. Keeping a continuous return proxy supports later diagnostics:

* Does high (p_{up}) concentrate higher realized returns?
* Are extreme moves assigned materially different probability mass?

### 6.3 Sanity checks

* Drop the last row (no (t+1))
* Manually spot-check rows by comparing `Close(t)` vs `Close(t+1)` and verifying `Target`
* Save: `data/sp500_data_with_target.csv`

---

## 7. Feature engineering (`feature_engineering.py` + `build_features.py`) — encoding market state

I built **15 interpretable baseline features** across five information dimensions:

1. Momentum
2. Trend
3. Volatility / risk regime
4. Volume / flow
5. Candle structure

### Design intent (QR perspective)

* **Trend**: MA(5/20) position and **MA_Slope_20** (trend acceleration proxy, not just level)
* **Volatility context**: ATR(14) to condition signals on risk regime
* **Flow**: OBV change (10-day) to capture participation/confirmation
* **Candle structure**: day-level OHLC geometry to proxy micro-structure patterns

### Engineering constraints

* Features use information up to (t) only
* Rolling windows naturally introduce NaNs at the start
  → handled by dropping rows with NaNs deterministically

### Output

* Keep: `Date, Close, Target` + 15 features
* Save: `data/features.csv`

### Research reflection

Early diagnostics suggested substantial redundancy/correlation across features. Instead of blindly adding more indicators, I treated this as a research signal: **manual indicators may have an expression ceiling** for next-day direction.

---

## 8. Time-aware validation (`split_sensitivity.py`) — avoiding “lucky windows”

Financial data is non-stationary; performance that depends on one particular time window is often spurious.

### What it does

Sensitivity analysis over test window length:

* Test = most recent **3 / 5 / 7 years**
* Train = all data before the test start

Baseline pipeline:

* `SimpleImputer(median) + StandardScaler + LogisticRegression(class_weight=balanced, max_iter=2000)`

### Why Logistic Regression as baseline

LR is a conservative benchmark (high bias, lower variance). If LR under strict time splits remains near-random, that supports the conclusion that the signal is genuinely weak rather than an implementation artifact.

### Output

* `results/split_sensitivity_results.csv` (if enabled in your implementation)

---

## 9. Walk-forward evaluation + regime analysis (`walk_forward_validation.py`)

A single split is insufficient for market data. I implemented **yearly expanding-window walk-forward** to approximate deployment.

### 9.1 Walk-forward setup

* Expanding training window
* Test each calendar year separately
* Model: `RandomForestClassifier(n_estimators=600, min_samples_leaf=3, class_weight=balanced_subsample)`

### 9.2 Metrics (why these)

Per year:

* Accuracy, Precision, Recall, F1 (classification summary)
* AUC (ranking/separation)
* LogLoss (probability quality; penalizes confident wrong predictions)
* Baseline Accuracy (majority/naive benchmark)

### 9.3 Regime labeling

Compute yearly `market_return` from first/last close of the year and label:

* Bull / Bear / Sideways

This is not for storytelling; it helps answer: **when does the model fail and why?**

### Output

* `results/walk_forward_year_metrics.csv`
* `results/walk_forward_predictions.csv`

---

## 10. Trading diagnostics (signal → strategy) — evaluating usefulness beyond accuracy

The probability output (p_{up}) is treated as a **ranking signal**, not merely a hard classifier.

### 10.1 Equity curve (`plot_equity_curve.py`)

Rule (default):

* Long if (p_{up} > 0.55), else flat
* Transaction cost: 2 bps

Outputs:

* `results/backtest_results.csv`
* `results/equity_curve.png` (if saved)

Purpose:

* study exposure vs turnover vs costs; not to claim “free alpha”

### 10.2 Decile returns (`plot_decile_returns.py`) — key ranking test

* Bucket (p_{up}) into 10 deciles
* Compute mean next-day return for each decile

Outputs:

* `results/decile_return_table.csv`
* `results/decile_returns.png`

Interpretation:

* If top deciles do not outperform bottom deciles, the model has limited ranking power (consistent with AUC ~ 0.5).

### 10.3 Threshold sweep (`threshold_sweep.py`)

* Sweep thresholds 0.50–0.70 step 0.01
* Track total return, max drawdown, exposure, Sharpe-like metrics

Outputs:

* `results/threshold_sweep.csv`
* `results/threshold_sweep.png`

Purpose:

* parameter sensitivity; avoid cherry-picking a single lucky threshold
* highlight the interaction between costs, selectivity, and exposure


## Plots (generated artifacts)

### Equity Curve (strategy vs buy-and-hold)
![Equity Curve](results/fig_equity_curve.png)

### Decile Returns (ranking diagnostic)
![Decile Returns](results/fig_decile_returns.png)

### Threshold Sweep (parameter sensitivity)
![Threshold Sweep](results/fig_threshold_sweep.png)

---

## 11. Post-mortem (engineering + research learnings)

### Engineering issues I faced and fixed

* **Path brittleness** → removed hard-coded paths, used `pathlib` for portable runs
* **Silent runs / no feedback** → added logging and artifact checks (CSV timestamps, row counts)
* **Schema mismatches** → centralized schema normalization in `data_cleaning.py`

### Research issues I clarified during the build

* Why random splits are invalid in finance (implicit leakage)
* Why accuracy is not the sole metric (AUC/LogLoss/ranking diagnostics matter)
* Why weak results are informative when the evaluation is strict

---

## 12. Why deep learning next (the “DL bridge”)

This project made the limits of traditional ML baselines tangible under strict evaluation:

1. **Time-dependence gap**
   RF/LR treat each day mostly as an independent sample; market dynamics often live in sequences.
2. **Manual feature ceiling**
   15 indicators are an interpretable projection, but may not capture cross-frequency non-linear structures.
3. **Regime instability**
   Walk-forward variance suggests sensitivity to market state shifts.

**Next step:** keep the same walk-forward evaluation “chassis” and swap the model “engine”:

* LSTM for gated memory of latent market states
* Transformer for long-range dependencies via attention
* Evaluate with the same diagnostics: walk-forward + deciles + out-of-sample threshold selection

---

## 13. How to run (high-level)

1. Install: `pip install -r requirements.txt`
2. Run scripts in order:

   * `data_collection.py` → `data_cleaning.py` → `make_target.py` → `feature_engineering.py` / `build_features.py`
   * `split_sensitivity.py` → `walk_forward_validation.py`
   * `plot_equity_curve.py` → `plot_decile_returns.py` → `threshold_sweep.py`
3. Check artifacts in `data/` and `results/`.

---

## 14. Disclaimer

This repository is for research/education. It is not financial advice. Past results do not guarantee future performance.

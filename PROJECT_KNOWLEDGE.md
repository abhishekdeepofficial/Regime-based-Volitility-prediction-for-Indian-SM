# PROJECT KNOWLEDGE — Hybrid Regime-Aware Multiscale Volatility Prediction System

## 1. PROJECT OVERVIEW

**Title:** Hybrid Regime-Aware Multiscale Volatility Prediction System for Indian Equity Markets

**Goal:** Build a system that predicts stock market volatility (Realized Volatility) at multiple time horizons using a combination of statistical models (GARCH, HAR-RV) and deep learning (Temporal Fusion Transformer / TFT), enhanced by market regime detection (HMM) and cross-asset features.

**Primary Asset:** Reliance Industries (NSE), with 13 other Indian blue-chip stocks for cross-sectional learning.

**Data Source:** 5-minute OHLCV bars for 14 stocks, spanning ~10 years (~194k bars per stock, ~2.65M total).

**Key Innovation:** Multi-asset trained TFT with regime-awareness, intraday seasonality adjustment, and GARCH conditional volatility as hybrid features.

---

## 2. DIRECTORY STRUCTURE

```
RResearch_Project/
├── config/
│   ├── settings.py              # Global paths, stock list, column schema
│   └── best_params.json         # Tuned TFT hyperparameters (Optuna output)
│
├── Data/                        # Raw 5-min CSV files (14 stocks)
│   ├── RELIANCE_5minute.csv
│   ├── HDFCBANK_5minute.csv
│   ├── INFY_5minute.csv
│   ├── TCS_5minute.csv
│   ├── SBIN_5minute.csv
│   ├── ICICIBANK_5minute.csv
│   ├── AXISBANK_5minute.csv
│   ├── LT_5minute.csv
│   ├── WIPRO_5minute.csv
│   ├── TATASTEEL_5minute.csv
│   ├── TITAN_5minute.csv
│   ├── COALINDIA_5minute.csv
│   ├── HAL_5minute.csv
│   ├── ADANIPORTS_5minute.csv
│   └── processed/               # Generated parquet files
│       ├── {SYMBOL}_processed.parquet   # Per-stock processed data
│       ├── pooled_data.parquet          # All stocks concatenated
│       ├── pooled_data_with_regimes.parquet  # + HMM regimes + DCC correlation
│       ├── pooled_data_with_garch.parquet    # + GARCH conditional volatility (FINAL)
│       ├── hmm_market_model.pkl         # Saved HMM regime detector
│       ├── market_correlation.csv       # DCC proxy output
│       └── market_regimes.csv           # Regime labels
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Loads raw CSVs for all stocks
│   │   ├── preprocessor.py         # Trading hours filter, bad tick removal, missing data
│   │   ├── feature_engineering.py   # Core features: RV, Jump, Range, Temporal, Volume
│   │   ├── pipeline.py             # Orchestrates load → preprocess → features → targets → pool
│   │   ├── attach_regimes.py       # Merges HMM regimes + DCC correlation to pooled data
│   │   └── add_garch_features.py   # Adds GARCH(1,1) conditional volatility per stock
│   │
│   ├── models/
│   │   ├── garch.py                # GARCH/EGARCH/GJR-GARCH model wrapper
│   │   ├── har.py                  # HAR-RV (Heterogeneous Autoregressive) model
│   │   ├── hmm_module.py           # RegimeDetector class (4-state Gaussian HMM)
│   │   ├── hmm.py                  # Runs regime detection on market-avg proxy
│   │   ├── dcc.py                  # DCC-GARCH proxy (rolling correlation index)
│   │   ├── run_baselines.py        # Trains GARCH+HAR-RV baselines for all stocks
│   │   ├── tft_dataset.py          # Prepares TimeSeriesDataSet for TFT
│   │   ├── train_tft.py            # TFT training script with checkpointing
│   │   ├── tune_tft.py             # Optuna hyperparameter tuning for TFT
│   │   └── evaluate_tft.py         # Evaluates TFT on validation set
│   │
│   ├── evaluation/
│   │   └── metrics.py              # RMSE, MAE, R², QLIKE, Directional Accuracy
│   │
│   └── analysis/
│       └── visualize_results.py    # Scatter plots, time series plots of predictions
│
├── run_pipeline.py                 # Master pipeline: 8 steps end-to-end
├── requirements.txt
├── baseline_results.csv            # GARCH/HAR-RV results for all 14 stocks
├── tft_evaluation_results.csv      # TFT evaluation metrics
├── lightning_logs/                  # TensorBoard training logs
├── checkpoints/                    # Saved model checkpoints
└── reports/                        # Generated figures
```

---

## 3. PIPELINE (run_pipeline.py) — 8 Steps

```
Step 1: Data Preprocessing & Pooling     → src/data/pipeline.py
Step 2: Market Regime Detection (HMM)    → src/models/hmm.py
Step 3: Market Correlation (DCC Proxy)   → src/models/dcc.py (via run_pipeline.run_dcc_step)
Step 4: Attach Regimes to Dataset        → src/data/attach_regimes.py
Step 5: Add GARCH Conditional Volatility → src/data/add_garch_features.py
Step 6: Train TFT                        → src/models/train_tft.py
Step 7: Evaluate Model                   → src/models/evaluate_tft.py
Step 8: Visualize Predictions            → src/analysis/visualize_results.py
```

**Data flow:**
```
Raw CSVs → [Step 1] → pooled_data.parquet (2.65M rows, 38 columns)
         → [Step 2] → hmm_market_model.pkl
         → [Step 3] → market_correlation.csv
         → [Step 4] → pooled_data_with_regimes.parquet (41 columns)
         → [Step 5] → pooled_data_with_garch.parquet (42 columns)
         → [Step 6] → TFT model checkpoint (.ckpt)
         → [Step 7] → tft_evaluation_results.csv
         → [Step 8] → reports/figures/*.png
```

---

## 4. STOCK LIST

Defined in `config/settings.py`:

```python
STOCKS = [
    'ADANIPORTS', 'AXISBANK', 'COALINDIA', 'HAL', 'HDFCBANK',
    'ICICIBANK', 'INFY', 'LT', 'RELIANCE', 'SBIN',
    'TATASTEEL', 'TCS', 'TITAN', 'WIPRO'
]
```

All are NSE-listed Indian large-caps. ~194k 5-minute bars each (~10 years of data).

---

## 5. DATA SCHEMA

### Raw CSV Columns
`date, open, high, low, close, volume`

### After Feature Engineering (47 columns in final parquet)

| Category | Columns | Description |
|----------|---------|-------------|
| **Price** | `open, high, low, close` | Raw OHLC |
| **Returns** | `log_price, log_return` | Log prices and returns |
| **Seasonality** | `time_idx, seasonal_factor, deseasonalized_return` | U-shape intraday pattern removal |
| **Realized Volatility** | `RV_1h, RV_half_day, RV_1d, RV_1w` | Multiscale RV (1h, 3h, 1 day, 1 week rolling windows) |
| **HAR-RV Lags** | `RV_1d_lag1, RV_1d_lag12, RV_1d_lag75, RV_1w_lag375, log_RV_1d` | Lagged RV features for autoregressive volatility capture |
| **Jump Detection** | `BV_1d, Jump_1d, Jump_ratio, circuit_breaker, Jump_sig` | Bipower variation, jump component, circuit breaker flags |
| **Range Estimators** | `Parkinson_1d, GK_1d, RS_1d` | Parkinson, Garman-Klass, Rogers-Satchell |
| **Temporal** | `hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_opening, is_closing` | Cyclical time encodings + session flags |
| **Volume** | `volume, volume_zscore, vwap_distance` | Volume z-score and VWAP distance |
| **Targets** | `target_RV_12bar, target_RV_48bar, target_RV_75bar` | Forward-looking RV at 1h, 4h, 1 day |
| **Regime** | `regime_id, regime` | HMM state: Low_Vol / Normal_Vol / High_Vol / Extreme_Vol |
| **Cross-Asset** | `market_correlation` | DCC proxy (rolling avg correlation with market) |
| **GARCH** | `garch_volatility` | GARCH(1,1) conditional volatility per stock |
| **Identifiers** | `symbol, symbol_id, datetime` | Stock symbol and timestamp |

### TFT Model Target

The TFT model uses `log_target_RV_75bar = log(1 + target_RV_75bar)` as the target — predicting next-day (75-bar) realized volatility. This is created dynamically in `tft_dataset.py`.

---

## 6. FEATURE ENGINEERING DETAILS

### 6.1 Intraday Seasonality Adjustment (src/data/feature_engineering.py)
- Computes U-shaped volatility profile by time-of-day
- `seasonal_factor = mean_abs_return(time) / overall_mean`
- `deseasonalized_return = log_return / seasonal_factor`

### 6.2 Realized Volatility
- `RV = sqrt(sum(r²))` over rolling windows
- Windows: 12 bars (1h), 38 bars (half-day), 75 bars (1 day), 375 bars (1 week)
- `RV_1d` is annualized: `× sqrt(252)`

### 6.3 Jump Detection (Barndorff-Nielsen & Shephard 2004)
- `BV_1d = sqrt((π/2) × sum(|r_t| × |r_{t-1}|) × 252)` (Bipower Variation)
- `Jump_1d = max(RV_1d - BV_1d, 0)`
- `Jump_ratio = Jump_1d / RV_1d`
- `circuit_breaker = 1` if `|return| > 9%` (India-specific)
- `Jump_sig = 1` if `Jump_ratio > 0.15` or circuit breaker hit

### 6.4 Range-Based Estimators
- **Parkinson:** `sqrt(mean(ln(H/L)²) / (4×ln(2)) × 252)`
- **Garman-Klass:** `sqrt(mean(0.5×ln(H/L)² - (2ln2-1)×ln(C/O)²) × 252)`
- **Rogers-Satchell:** `sqrt(mean(ln(H/C)×ln(H/O) + ln(L/C)×ln(L/O)) × 252)`

### 6.5 Temporal Encoding
- Cyclical: `sin/cos(2π × hour/24)`, `sin/cos(2π × dow/5)`, `sin/cos(2π × month/12)`
- Session flags: `is_opening` (first 30 min), `is_closing` (last 30 min)

---

## 7. MODELS

### 7.1 GARCH Family (src/models/garch.py)
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1,1)
- Student-t distribution
- Returns scaled by ×100 for numerical stability
- Best model selected by AIC per stock

### 7.2 HAR-RV (src/models/har.py)
- Heterogeneous Autoregressive model of Realized Volatility
- Lags: daily (1), weekly (5), monthly (22)
- OLS with Newey-West (HAC) standard errors

### 7.3 HMM Regime Detection (src/models/hmm_module.py)
- 4-state Gaussian HMM (hmmlearn)
- Features: market-average `log_return` and `RV_1d` (scaled ×100)
- States ordered by volatility: Low_Vol → Normal_Vol → High_Vol → Extreme_Vol

### 7.4 DCC Proxy (src/models/dcc.py)
- Fits GARCH(1,1) to each stock → standardized residuals
- Rolling correlation of each stock's residuals with equal-weighted market return
- Window: 375 bars (1 week)

### 7.5 Temporal Fusion Transformer (src/models/tft_dataset.py + train_tft.py)

**Model A** (with autoregressive input):
- **Architecture:** 1.9M parameters
  - hidden_size=128, attention_head_size=4, dropout=0.15
  - hidden_continuous_size=64, output_size=1 (point prediction)
- **Training:** 100 epochs max, early stopping patience=15, gradient_clip=0.1, LR=0.001
- **Batch limits:** 1000 train batches, 200 val batches per epoch

**Model B v2** (without autoregressive input — Kaggle GPU training):
- **Architecture:** 7.3M parameters (3.8× larger than Model A)
  - hidden_size=256, attention_head_size=8, dropout=0.1
  - hidden_continuous_size=128, output_size=1
- **Training:** 150 epochs max, early stopping patience=20, gradient_clip=0.05, LR=0.0005
- **Batch limits:** 1500 train batches, 300 val batches per epoch
- **Training script:** `kaggle_train_model_b.py` (self-contained for Kaggle)

**Common settings for both models:**
- **Target:** `log_target_RV_75bar = log(1 + target_RV_75bar)` — next-day RV
- **Target normalizer:** GroupNormalizer per symbol with softplus transformation
- **Loss:** RMSE
- **Encoder length:** 75 bars (1 day lookback)
- **Prediction length:** 1 step

**TFT Input Features:**

| Type | Model A | Model B |
|------|---------|--------|
| **Static categorical** | `symbol` | `symbol` |
| **Known future reals (7)** | `time_idx_global, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos` | Same |
| **Known future cats (2)** | `is_opening, is_closing` | Same |
| **Unknown reals** | 21 (incl. `log_target_RV_75bar`) | 20 (excl. target) |
| **Unknown cats (1)** | `regime` | `regime` |

---

## 8. EVALUATION METRICS (src/evaluation/metrics.py)

| Metric | Formula | Notes |
|--------|---------|-------|
| RMSE | `sqrt(mean((y-ŷ)²))` | Lower is better |
| MAE | `mean(|y-ŷ|)` | Lower is better |
| R² | `1 - SS_res/SS_tot` | Higher is better, >0 means beats mean |
| QLIKE | `mean(log(ŷ) + y/ŷ)` | Standard volatility forecast metric, lower is better |
| DA | `mean(sign(Δy) == sign(Δŷ))` | Directional accuracy, higher is better |
| MAPE | `mean(|y-ŷ|/y) × 100` | Percentage error |

---

## 9. BASELINE RESULTS (14 stocks)

| Stock | Best GARCH | GARCH AIC | HAR-RV R² |
|-------|-----------|-----------|-----------|
| ADANIPORTS | GARCH | -83,397 | 0.994 |
| AXISBANK | GARCH | -157,556 | 0.994 |
| COALINDIA | GJR-GARCH | 217,378 | 0.991 |
| HAL | GJR-GARCH | 221,094 | 0.990 |
| HDFCBANK | GARCH | -288,868 | 0.993 |
| ICICIBANK | GARCH | 571,365 | 0.992 |
| INFY | GJR-GARCH | -218,151 | 0.981 |
| LT | GARCH | 94,622 | 0.991 |
| RELIANCE | GARCH | 2,627,529 | 0.993 |
| SBIN | EGARCH | 1,667,872 | 0.993 |
| TATASTEEL | GARCH | 1,287,671 | 0.992 |
| TCS | EGARCH | -272,294 | 0.990 |
| TITAN | GARCH | 199,085 | 0.990 |
| WIPRO | GJR-GARCH | 332,889 | 0.988 |

HAR-RV R² > 0.98 for all stocks (strong baseline using lagged RV).

---

## 10. CONFIGURATION (config/settings.py)

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"                    # Raw CSVs
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"  # Note: lowercase "data"
COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
```

**Tuned Hyperparams (config/best_params.json):**
```json
{
    "learning_rate": 0.001,
    "hidden_size": 128,
    "dropout": 0.15,
    "hidden_continuous_size": 64,
    "attention_head_size": 4
}
```

---

## 11. KEY DESIGN DECISIONS & KNOWN ISSUES

1. **Log-transformed target:** `log(1 + RV)` used because raw RV can be very large (annualized), causing numerical instability.
2. **GroupNormalizer with softplus:** Ensures positive predictions per-stock.
3. **Corporate actions adjustment:** Currently a placeholder/stub in `preprocessor.py` (not implemented).
4. **Missing data handling:** `handle_missing_data()` in `preprocessor.py` is implemented but commented out in the `process()` method — skipped for speed.
5. **DCC is a proxy:** True DCC-GARCH is expensive; the implementation uses rolling correlation of GARCH standardized residuals as an approximation.
6. **Market hours:** Indian market 9:15 AM – 3:30 PM IST = 75 five-minute bars per day, 250 trading days/year.
7. **GARCH on 5-min data:** Full-sample GARCH fit (not rolling window), which means parameters are constant but conditional volatility is still dynamic.

---

## 12. DEPENDENCIES

```
pytorch-forecasting
pytorch-lightning (lightning)
torch
pandas, numpy
arch (GARCH models)
hmmlearn (HMM regime detection)
statsmodels (HAR-RV, OLS)
scikit-learn (metrics)
matplotlib, seaborn (visualization)
optuna (hyperparameter tuning)
tqdm, joblib
```

---

## 13. HOW TO RUN

```bash
# Full pipeline (all 8 steps):
cd "/Users/abhishekdeep/Documents/Data/MTech/Project-Sem 2/RResearch_Project"
python3 run_pipeline.py

# Individual steps:
python3 -m src.data.pipeline           # Step 1: Data processing
python3 -m src.models.hmm              # Step 2: Regime detection
python3 -m src.models.dcc              # Step 3: DCC correlation
python3 -m src.data.attach_regimes     # Step 4: Merge regimes
python3 -m src.data.add_garch_features # Step 5: GARCH features
python3 -m src.models.train_tft        # Step 6: Train TFT (Model A — with AR input)
python3 -m src.models.evaluate_tft     # Step 7: Evaluate
python3 -m src.analysis.visualize_results  # Step 8: Plots

# Ablation study (Model B — without autoregressive input):
python3 src/models/train_model_b.py
python3 src/analysis/compare_models.py

# Baselines only:
python3 -m src.models.run_baselines

# Hyperparameter tuning:
python3 -m src.models.tune_tft
```

---

## 14. LEAKAGE & OVERLAP AUDIT

### Target Construction (src/data/pipeline.py)
- `target_RV_75bar = sqrt(sum(r²[t+1 ... t+75])) × sqrt(252)`
- Uses **only future bars** (forward-looking) — zero overlap with backward-looking RV features ✅

### Autoregressive Input Overlap
- With `max_prediction_length=1`, consecutive targets share **74/75 bars = 98.7%** overlap
- Target autocorrelation at lag-1: **0.9963** (very high persistence)
- This is a known characteristic of rolling-window RV — not data leakage, but the model can exploit it
- **Mitigation:** Ablation study compares with/without autoregressive input

### Feature-Target Overlap
- `RV_1d` (backward rolling 75 bars `[t-74..t]`) vs `target_RV_75bar` (forward `[t+1..t+75]`) → **0 bars overlap** ✅
- `RV_1d_lag1` (shifted by 1 bar) → **0 bars overlap** ✅
- All HAR-RV lag features: **0 overlap** ✅

---

## 15. ABLATION STUDY SETUP

| | Model A | Model B v1 | Model B v2 |
|---|---------|-----------|------------|
| **AR input** | ✅ Included | ❌ Excluded | ❌ Excluded |
| **hidden_size** | 128 | 128 | 256 |
| **attention_heads** | 4 | 4 | 8 |
| **Parameters** | 1.9M | 1.9M | 7.3M |
| **LR / Dropout** | 0.001 / 0.15 | 0.001 / 0.15 | 0.0005 / 0.1 |
| **R²** | 0.976 | 0.258 | *Training...* |
| **Script** | `train_tft.py` | `train_model_b.py` | `kaggle_train_model_b.py` |
| **Checkpoint** | `checkpoints/tft/` | `checkpoints/tft_model_b/` | Kaggle Output |

The `use_autoregressive` flag in `tft_dataset.py` controls which features are included.

### Ablation Results (v1)

| Metric | Model A (with AR) | Model B v1 (w/o AR) | Δ |
|--------|-------------------|--------------------|---------|
| R² | **0.976** | 0.258 | −0.718 |
| RMSE | **0.009** | 0.052 | +0.043 |
| MAPE | **1.90%** | 19.26% | +17.4pp |

Model B v2 training in progress on Kaggle (T4 GPU) with improved architecture. Expected completion: Feb 23, 2026.

---

## 16. VISUALIZATION SYSTEM (src/analysis/visualize_results.py)

Generates 3 types of plots:
1. **Scatter plot**: Predicted vs Actual daily RV across all stocks (with overall R²)
2. **Per-stock 2-panel timeline**: Top panel shows last 6 months of training (gray) + validation actual (black) + predicted (blue) with train/val split line. Bottom panel shows validation zoom with error shading.
3. **R² bar chart**: Per-stock R² color-coded (green ≥0.3, orange ≥0.1, red <0.1) with mean R² line.

All plots use real timestamps on x-axis and are saved to `reports/figures/`.

---

## 17. KAGGLE CLOUD TRAINING

Model B v2 is trained on Kaggle (free T4 GPU) since local MPS training was too slow (~3 min/batch).

**Setup:**
1. Data: `pooled_data_with_garch.parquet` uploaded as Kaggle Dataset "Volitility-data"
2. Code: `kaggle_train_model_b.py` self-contained script (cloned from GitHub)
3. GPU: Tesla T4 (15.6 GB), ~1.78 it/s, ~14 min/epoch

**Workflow:**
```bash
!git clone https://github.com/abhishekdeepofficial/Regime-based-Volitility-prediction-for-Indian-SM.git /kaggle/working/project
!cd /kaggle/working/project && python kaggle_train_model_b.py
```

**After training:** Download `.ckpt` from Kaggle Output tab → `checkpoints/tft_model_b/`

---

## 18. CURRENT STATUS (Feb 23, 2026)

- **Data pipeline:** Fully working, 2.65M rows with 47 features across 14 stocks
- **Model A (with AR):** ✅ Complete — R²=0.976, RMSE=0.009, MAPE=1.90%
- **Model B v1 (w/o AR):** ✅ Complete — R²=0.258, RMSE=0.052, MAPE=19.26%
- **Model B v2 (w/o AR, improved):** 🔄 Training on Kaggle T4 GPU (7.3M params, hidden=256, heads=8)
  - Epoch 0 val_loss: 0.042, Epoch 1 val_loss: 0.043
  - Expected completion: ~6-8 hours from start
- **Regime analysis:** All 4 regimes covered (Low/Normal/High/Extreme) with expanded 5,000-step validation
- **Leakage audit:** ✅ Feature-target overlap zero; AR overlap disclosed and mitigated via ablation
- **Research paper:** Springer LNCS format (`paper_springer/paper.tex`), drafts in `research_paper.md`
- **GitHub:** [abhishekdeepofficial/Regime-based-Volitility-prediction-for-Indian-SM](https://github.com/abhishekdeepofficial/Regime-based-Volitility-prediction-for-Indian-SM)
- **Pending:** Update paper with Model B v2 final results after Kaggle training completes

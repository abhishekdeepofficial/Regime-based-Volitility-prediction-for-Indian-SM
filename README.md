# Hybrid Regime-Aware Multiscale Volatility Prediction System

An advanced Deep Learning framework for forecasting intraday realized volatility across multiple assets in the Indian Stock Market (NSE). This project utilizes a **Regime-Aware Temporal Fusion Transformer (TFT)** to capture non-linear dependencies and structural market breaks.

## 🚀 Features
- **Multi-Asset Support**: Trained on 14 liquid NSE stocks (RELIANCE, HDFCBANK, TCS, etc.).
- **Regime Awareness**: Integrates Hidden Markov Model (HMM) derived market regimes.
- **Deep Learning**: Uses Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting.
- **Robustness**: Achieves **0.89 R²** and **92% Directional Accuracy** for 5-minute volatility prediction.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch (with MPS support for Mac or CUDA for NVIDIA GPUs)

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Usage Pipeline

Follow these steps to reproduce the results from raw data to final evaluation.

### 1. Data Engineering Pipeline
Run the following scripts in order to process raw OHLCV data into a clean, regime-enriched dataset.

```bash
# A. Load, Clean, and Feature Engineer Data (Output: pooled_data.parquet)
python3 src/data/pipeline.py

# B. Train Regime Detection Model (HMM) & Attach Regimes (Output: pooled_data_with_regimes.parquet)
python3 src/data/attach_regimes.py
```

### 2. Baseline Models (Optional)
Train and evaluate statistical baselines (GARCH, HAR-RV).

```bash
python3 src/models/run_baselines.py
```

### 3. Train Deep Learning Model (TFT)
Train the Temporal Fusion Transformer on the processed multi-asset dataset.
*Note: The script automatically uses the best hyperparameters found during tuning.*

```bash
# Train the Regime-Aware TFT
python3 src/models/train_tft.py
```

### 4. Evaluation
Evaluate the trained model on the held-out test set and generate metrics.

```bash
# Generate metrics (RMSE, R2, QLIKE, DA)
python3 src/models/evaluate_tft.py
```

## 📊 Results Summary

| Model | Horizon | R² | Directional Accuracy |
| :--- | :--- | :--- | :--- |
| **Regime-Aware TFT** | **5-min** | **0.89** | **92.3%** |
| HAR-RV (Baseline) | 5-min | ~0.65 | ~58% |

## 📂 Project Structure
- `src/data/`: Data loading, cleaning, and feature engineering scripts.
- `src/models/`: Model implementations (TFT, GARCH, HMM).
- `config/`: Configuration settings and hyperparameter files.
- `notebooks/`: Exploratory analysis.

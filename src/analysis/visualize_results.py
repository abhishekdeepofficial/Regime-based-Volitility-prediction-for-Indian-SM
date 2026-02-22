import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.metrics import r2_score

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.settings import PROCESSED_DATA_DIR, STOCKS
from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)


def find_best_checkpoint():
    """Find best checkpoint from training."""
    ckpt_dir = project_root / "checkpoints" / "tft"
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime) if ckpt_dir.exists() else []
    if not checkpoints:
        log_dir = project_root / "lightning_logs"
        if log_dir.exists():
            checkpoints = sorted(log_dir.rglob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def visualize_predictions():
    """
    Generate comprehensive visualizations showing:
    1. Full timeline per stock: Train data | Validation data + Predictions with real timestamps
    2. Overall scatter plot of predicted vs actual
    3. Per-stock R² in titles
    """
    print("Loading data and model...")
    data = load_and_prep_tft_data()
    train_ds, val_ds = create_tft_dataset(data)

    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Find and load model
    best_model_path = find_best_checkpoint()
    if best_model_path is None:
        print("No checkpoints found.")
        return

    print(f"Loading model from {best_model_path}")
    tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Generate predictions on validation set
    print("Generating predictions...")
    raw_predictions = tft.predict(val_dl, return_y=True, return_x=True)
    y_pred = raw_predictions.output.cpu().numpy()   # (N, 1)
    y_true = raw_predictions.y[0].cpu().numpy()      # (N, 1)
    x = raw_predictions.x

    # Get time indices and group info
    decoder_time_idx = x['decoder_time_idx'].cpu().numpy()  # (N, pred_len)
    groups = x['groups'].cpu().numpy().flatten()              # (N,)
    symbol_encoder = val_ds.categorical_encoders['symbol']

    # Build time_idx -> datetime mapping from original data
    time_idx_to_dt = dict(zip(data['time_idx_global'], data['datetime']))

    # Get the training cutoff time_idx
    max_prediction_length = val_ds.max_prediction_length
    training_cutoff = data['time_idx_global'].max() - max_prediction_length - 1000

    report_dir = project_root / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ========== PLOT 1: Overall Scatter Plot ==========
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    overall_r2 = r2_score(y_true_flat, y_pred_flat)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.05, s=2, color='steelblue')
    mn, mx = y_true_flat.min(), y_true_flat.max()
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Daily RV (Log-Scaled)", fontsize=12)
    plt.ylabel("Predicted Daily RV (Log-Scaled)", fontsize=12)
    plt.title(f"TFT: Next-Day RV Prediction — All Stocks (R²={overall_r2:.3f})", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(report_dir / "scatter_pred_vs_actual_daily.png", dpi=150)
    plt.close()
    print(f"  Saved scatter plot (R²={overall_r2:.3f})")

    # ========== PLOT 2: Per-Stock Full Timeline ==========
    # For each stock, show:
    #   - Train period: actual target (gray)
    #   - Validation period: actual target (black) + predicted (blue)
    #   - Vertical line at train/val split

    target_col = 'log_target_RV_75bar'

    for stock in STOCKS:
        if stock not in symbol_encoder.classes_:
            continue

        # --- Full timeline from raw data ---
        stock_data = data[data['symbol'] == stock].sort_values('time_idx_global').copy()

        if len(stock_data) == 0 or target_col not in stock_data.columns:
            continue

        # Split into train and validation based on cutoff
        train_data = stock_data[stock_data['time_idx_global'] <= training_cutoff]
        val_data = stock_data[stock_data['time_idx_global'] > training_cutoff]

        # --- Get predictions for this stock ---
        stock_idx = symbol_encoder.transform([stock])[0]
        pred_indices = np.where(groups == stock_idx)[0]

        if len(pred_indices) == 0:
            continue

        # Map predictions to datetimes
        pred_time_idxs = decoder_time_idx[pred_indices, 0]  # First (only) prediction step
        pred_datetimes = [time_idx_to_dt.get(int(tid)) for tid in pred_time_idxs]
        pred_values = y_pred[pred_indices, 0]
        actual_values = y_true[pred_indices, 0]

        # Filter out None datetimes
        valid = [i for i, dt in enumerate(pred_datetimes) if dt is not None]
        if not valid:
            continue
        pred_datetimes = [pred_datetimes[i] for i in valid]
        pred_values = pred_values[valid]
        actual_values = actual_values[valid]

        # Compute per-stock R²
        stock_r2 = r2_score(actual_values, pred_values) if len(actual_values) > 1 else 0
        stock_mae = np.mean(np.abs(actual_values - pred_values))

        # --- Create the plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top: Full timeline
        # Train period — show last 6 months for context
        if len(train_data) > 0 and 'datetime' in train_data.columns:
            train_daily = train_data.groupby(train_data['datetime'].dt.date)[target_col].mean()
            # Only show last 180 days of training for context
            train_tail = train_daily.iloc[-180:] if len(train_daily) > 180 else train_daily
            ax1.plot(pd.to_datetime(train_tail.index), train_tail.values,
                     color='gray', alpha=0.5, linewidth=0.8, label='Train (last 6 months)')

        # Validation actuals
        if len(val_data) > 0 and 'datetime' in val_data.columns:
            val_daily = val_data.groupby(val_data['datetime'].dt.date)[target_col].mean()
            ax1.plot(pd.to_datetime(val_daily.index), val_daily.values,
                     color='black', alpha=0.8, linewidth=1.0, label='Validation (actual)')

        # Predictions (aggregate to daily for clean plotting)
        pred_df = pd.DataFrame({
            'datetime': pred_datetimes,
            'predicted': pred_values,
            'actual': actual_values
        })
        pred_df['date'] = pd.to_datetime(pred_df['datetime']).dt.date
        pred_daily = pred_df.groupby('date').agg({'predicted': 'mean', 'actual': 'mean'})

        ax1.plot(pd.to_datetime(pred_daily.index), pred_daily['predicted'].values,
                 color='dodgerblue', alpha=0.9, linewidth=1.2, label='TFT Predicted')

        # Train/Val split line
        if len(val_data) > 0 and len(train_data) > 0:
            split_date = val_data['datetime'].min()
            ax1.axvline(x=split_date, color='red', linestyle='--', alpha=0.7, label='Train/Val Split')

        ax1.set_title(f"{stock}: Daily RV — Full Timeline (R²={stock_r2:.3f}, MAE={stock_mae:.4f})",
                       fontsize=14, fontweight='bold')
        ax1.set_ylabel("Log Daily RV", fontsize=11)
        ax1.legend(loc='upper left', fontsize=9, ncol=4)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Bottom: Validation zoom — actual vs predicted
        ax2.plot(pd.to_datetime(pred_daily.index), pred_daily['actual'].values,
                 color='black', linewidth=1.0, label='Actual')
        ax2.plot(pd.to_datetime(pred_daily.index), pred_daily['predicted'].values,
                 color='dodgerblue', linewidth=1.0, label='Predicted')
        ax2.fill_between(pd.to_datetime(pred_daily.index),
                         pred_daily['actual'].values,
                         pred_daily['predicted'].values,
                         alpha=0.2, color='orange', label='Error')

        ax2.set_title(f"Validation Period: Actual vs Predicted", fontsize=12)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Log Daily RV", fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / f"timeseries_{stock}.png", dpi=150)
        plt.close()
        print(f"  Saved {stock} (R²={stock_r2:.3f})")

    # ========== PLOT 3: Summary Bar Chart — R² per stock ==========
    stock_r2_values = {}
    for stock in STOCKS:
        if stock not in symbol_encoder.classes_:
            continue
        stock_idx = symbol_encoder.transform([stock])[0]
        pred_indices = np.where(groups == stock_idx)[0]
        if len(pred_indices) > 1:
            r2 = r2_score(y_true[pred_indices, 0], y_pred[pred_indices, 0])
            stock_r2_values[stock] = r2

    if stock_r2_values:
        fig, ax = plt.subplots(figsize=(14, 6))
        stocks = list(stock_r2_values.keys())
        r2_vals = list(stock_r2_values.values())
        colors = ['forestgreen' if r > 0.3 else 'orange' if r > 0.1 else 'red' for r in r2_vals]

        bars = ax.bar(stocks, r2_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=np.mean(r2_vals), color='blue', linestyle='--', label=f'Mean R²={np.mean(r2_vals):.3f}')
        ax.set_xlabel("Stock", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("Per-Stock R² — Next-Day RV Prediction", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(report_dir / "r2_per_stock.png", dpi=150)
        plt.close()
        print(f"  Saved R² per stock chart")

    print(f"\nAll plots saved to {report_dir}")


if __name__ == "__main__":
    visualize_predictions()

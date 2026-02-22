"""
Visualize Model B predictions (without autoregressive input).
Same plots as visualize_results.py but for the ablation Model B.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pytorch_forecasting import TemporalFusionTransformer

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def visualize_model_b():
    print("=" * 60)
    print("  MODEL B VISUALIZATION (without autoregressive input)")
    print("=" * 60)

    # Load data — Model B uses use_autoregressive=False
    data = load_and_prep_tft_data()
    train_ds, val_ds = create_tft_dataset(data, use_autoregressive=False)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Find best Model B checkpoint
    ckpt_dir = project_root / "checkpoints" / "tft_model_b"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    if not ckpts:
        print("No Model B checkpoint found. Train it first.")
        return
    best_ckpt = ckpts[-1]
    print(f"Loading: {best_ckpt.name}")

    tft = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
    outputs = tft.predict(val_dl, return_y=True, return_x=True)

    y_pred = outputs.output.cpu().numpy()
    y_true = outputs.y[0].cpu().numpy() if isinstance(outputs.y, (list, tuple)) else outputs.y.cpu().numpy()

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    h_idx = 0
    report_dir = project_root / "reports" / "figures" / "model_b"
    report_dir.mkdir(parents=True, exist_ok=True)

    target_col = 'log_target_RV_75bar'

    # ========== Plot 1: Overall Scatter ==========
    overall_r2 = r2_score(y_true[:, h_idx], y_pred[:, h_idx])

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true[:, h_idx], y_pred[:, h_idx], alpha=0.05, s=2, color='darkorange')
    mn, mx = y_true[:, h_idx].min(), y_true[:, h_idx].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Actual Daily RV (Log-Scaled)")
    plt.ylabel("Predicted Daily RV (Log-Scaled)")
    plt.title(f"Model B (No AR): Next-Day RV Prediction — All Stocks (R²={overall_r2:.3f})")
    plt.tight_layout()
    plt.savefig(report_dir / "scatter_pred_vs_actual_daily.png", dpi=150)
    plt.close()
    print(f"  Saved scatter plot (R²={overall_r2:.3f})")

    # ========== Plot 2: Per-Stock Timeseries ==========
    x_data = outputs.x
    groups = x_data['groups'].cpu().numpy().flatten()
    decoder_time_idx = x_data['decoder_time_idx'].cpu().numpy()[:, 0]

    symbol_encoder = val_ds.categorical_encoders['symbol']
    stock_names = {}
    for cls in symbol_encoder.classes_:
        stock_names[symbol_encoder.transform([cls])[0]] = cls

    # Map time_idx to datetime
    time_idx_to_dt = dict(zip(data['time_idx_global'], data['datetime']))

    per_stock_r2 = {}
    for stock_id, stock_name in sorted(stock_names.items()):
        mask = groups == stock_id
        if mask.sum() < 10:
            continue

        s_true = y_true[mask, h_idx]
        s_pred = y_pred[mask, h_idx]
        s_tidx = decoder_time_idx[mask].astype(int)
        s_dates = [time_idx_to_dt.get(t) for t in s_tidx]

        stock_r2 = r2_score(s_true, s_pred)
        per_stock_r2[stock_name] = stock_r2

        # Get training data for context
        stock_data = data[data['symbol'] == stock_name].sort_values('time_idx_global')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Train data — last 6 months
        if 'datetime' in stock_data.columns and target_col in stock_data.columns:
            train_data = stock_data[stock_data['time_idx_global'] < val_ds.min_prediction_idx - val_ds.max_encoder_length]
            if len(train_data) > 0:
                train_daily = train_data.groupby(train_data['datetime'].dt.date)[target_col].mean()
                train_tail = train_daily.iloc[-180:] if len(train_daily) > 180 else train_daily
                ax1.plot(pd.to_datetime(train_tail.index), train_tail.values,
                         color='gray', alpha=0.5, linewidth=0.8, label='Train (last 6 months)')

        # Validation: actual + predicted
        valid_mask = [d is not None for d in s_dates]
        s_dates_valid = [d for d, v in zip(s_dates, valid_mask) if v]
        s_true_valid = s_true[valid_mask]
        s_pred_valid = s_pred[valid_mask]

        if len(s_dates_valid) > 0:
            dates_dt = pd.to_datetime(s_dates_valid)
            sort_idx = np.argsort(dates_dt)
            dates_sorted = dates_dt[sort_idx]
            true_sorted = s_true_valid[sort_idx]
            pred_sorted = s_pred_valid[sort_idx]

            # Daily aggregation
            df_plot = pd.DataFrame({'date': dates_sorted, 'true': true_sorted, 'pred': pred_sorted})
            df_daily = df_plot.groupby(df_plot['date'].dt.date).mean()

            ax1.plot(pd.to_datetime(df_daily.index), df_daily['true'].values,
                     color='black', linewidth=1.2, label='Actual', alpha=0.8)
            ax1.plot(pd.to_datetime(df_daily.index), df_daily['pred'].values,
                     color='darkorange', linewidth=1.0, label='Predicted (Model B)', alpha=0.8)

            # Error panel
            errors = df_daily['pred'].values - df_daily['true'].values
            ax2.fill_between(pd.to_datetime(df_daily.index), errors, 0,
                             where=errors >= 0, color='darkorange', alpha=0.3, label='Over-predict')
            ax2.fill_between(pd.to_datetime(df_daily.index), errors, 0,
                             where=errors < 0, color='purple', alpha=0.3, label='Under-predict')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.set_ylabel("Prediction Error")
            ax2.legend(fontsize=8)

        ax1.set_title(f"{stock_name} — Model B (No AR) — R²={stock_r2:.3f}", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Log Daily RV")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir / f"timeseries_{stock_name}.png", dpi=150)
        plt.close()
        print(f"  Saved {stock_name} (R²={stock_r2:.3f})")

    # ========== Plot 3: R² Per Stock Bar Chart ==========
    if per_stock_r2:
        fig, ax = plt.subplots(figsize=(14, 6))
        stocks = list(per_stock_r2.keys())
        r2_vals = list(per_stock_r2.values())
        colors = ['#2ecc71' if r >= 0.3 else '#e67e22' if r >= 0.1 else '#e74c3c' for r in r2_vals]

        bars = ax.bar(stocks, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=np.mean(r2_vals), color='blue', linestyle='--', linewidth=1.5,
                   label=f'Mean R²={np.mean(r2_vals):.3f}')
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("Model B (No AR): Per-Stock R²", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xticklabels(stocks, rotation=45, ha='right')

        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(report_dir / "r2_per_stock.png", dpi=150)
        plt.close()
        print(f"  Saved R² per stock chart")

    print(f"\nAll Model B plots saved to {report_dir}")


if __name__ == "__main__":
    visualize_model_b()

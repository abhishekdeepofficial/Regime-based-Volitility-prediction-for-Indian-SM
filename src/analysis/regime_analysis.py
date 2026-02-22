"""
Regime-Specific Performance Analysis
Shows how the TFT performs in different market regimes (Low/Normal/High/Extreme volatility).
This is a key differentiator for the paper — demonstrates adaptive regime-aware prediction.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pytorch_forecasting import TemporalFusionTransformer

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def find_best_checkpoint():
    ckpt_dir = project_root / "checkpoints" / "tft"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime) if ckpt_dir.exists() else []
    if not ckpts:
        log_dir = project_root / "lightning_logs"
        if log_dir.exists():
            ckpts = sorted(log_dir.rglob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def regime_analysis():
    print("=" * 60)
    print("  REGIME-SPECIFIC PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Load data and model
    # Use wider evaluation window (5000 time steps) to capture ALL 4 regimes
    data = load_and_prep_tft_data()
    
    max_enc = 75
    max_pred = 1
    eval_window = 5000  # much wider than default 1000 to include High_Vol regime
    
    target = 'log_target_RV_75bar'
    time_varying_unknown_reals = [
        'log_return', 'RV_1h', 'RV_half_day', 'RV_1d', 'RV_1w',
        'volume_zscore', 'market_correlation', target,
    ]
    optional_unknown = [
        'garch_volatility', 'Jump_1d', 'Jump_ratio',
        'Parkinson_1d', 'GK_1d', 'RS_1d', 'BV_1d', 'vwap_distance',
        'RV_1d_lag1', 'RV_1d_lag12', 'RV_1d_lag75', 'RV_1w_lag375', 'log_RV_1d',
    ]
    for col in optional_unknown:
        if col in data.columns:
            time_varying_unknown_reals.append(col)
    
    time_varying_known_reals = ['time_idx_global']
    for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']:
        if col in data.columns:
            time_varying_known_reals.append(col)
    
    time_varying_known_categoricals = []
    for col in ['is_opening', 'is_closing']:
        if col in data.columns:
            data[col] = data[col].astype(str)
            time_varying_known_categoricals.append(col)
    
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    
    training_cutoff = data['time_idx_global'].max() - max_pred - eval_window
    
    train_ds = TimeSeriesDataSet(
        data[data['time_idx_global'] <= training_cutoff],
        time_idx="time_idx_global",
        target=target,
        group_ids=["symbol"],
        min_encoder_length=max_enc // 2,
        max_encoder_length=max_enc,
        min_prediction_length=1,
        max_prediction_length=max_pred,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=['regime'],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, data, predict=False, stop_randomization=True,
        min_prediction_idx=training_cutoff + 1,
        min_prediction_length=max_pred,
    )
    
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # Check regime coverage
    val_data = data[data['time_idx_global'] > training_cutoff]
    print(f"\n  Evaluation window: {eval_window} time steps")
    print(f"  Regime distribution in eval set:")
    for r in ['Low_Vol', 'Normal_Vol', 'High_Vol', 'Extreme_Vol']:
        n = (val_data['regime'] == r).sum()
        print(f"    {r}: {n:,}")

    best_ckpt = find_best_checkpoint()
    if not best_ckpt:
        print("No checkpoint found.")
        return
    print(f"Loading model: {best_ckpt.name}")

    tft = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
    outputs = tft.predict(val_dl, return_y=True, return_x=True)

    y_pred = outputs.output.cpu().numpy().flatten()
    y_true = outputs.y[0].cpu().numpy().flatten()
    x = outputs.x

    # Get regime info from decoder
    groups = x['groups'].cpu().numpy().flatten()
    decoder_time_idx = x['decoder_time_idx'].cpu().numpy()[:, 0]

    # Map predictions back to regime labels via time_idx
    time_idx_to_regime = dict(zip(data['time_idx_global'], data['regime']))
    time_idx_to_symbol = dict(zip(data['time_idx_global'], data['symbol']))
    symbol_encoder = val_ds.categorical_encoders['symbol']

    # Build prediction DataFrame
    pred_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'time_idx': decoder_time_idx.astype(int),
    })
    pred_df['regime'] = pred_df['time_idx'].map(time_idx_to_regime)
    pred_df['stock_idx'] = groups

    # Decode stock symbols
    stock_names = {}
    for cls in symbol_encoder.classes_:
        stock_names[symbol_encoder.transform([cls])[0]] = cls
    pred_df['symbol'] = pred_df['stock_idx'].map(stock_names)

    # Drop rows without regime mapping
    pred_df = pred_df.dropna(subset=['regime'])

    report_dir = project_root / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ========== TABLE 1: Metrics per regime ==========
    print("\n" + "=" * 60)
    print("  Metrics by Market Regime")
    print("=" * 60)

    regime_order = ['Low_Vol', 'Normal_Vol', 'High_Vol', 'Extreme_Vol']
    regime_metrics = []

    for regime in regime_order:
        subset = pred_df[pred_df['regime'] == regime]
        if len(subset) < 10:
            continue
        r2 = r2_score(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        mape = np.mean(np.abs((subset['y_true'] - subset['y_pred']) / (subset['y_true'] + 1e-8))) * 100
        n = len(subset)

        regime_metrics.append({
            'Regime': regime,
            'Samples': n,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
        })
        print(f"  {regime:15s}  N={n:5d}  R²={r2:.4f}  RMSE={rmse:.6f}  MAPE={mape:.2f}%")

    metrics_df = pd.DataFrame(regime_metrics)
    metrics_df.to_csv(project_root / "regime_performance.csv", index=False)
    print(f"\nSaved regime_performance.csv")

    # ========== PLOT 1: R² by Regime (Bar Chart) ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {'Low_Vol': '#2ecc71', 'Normal_Vol': '#3498db', 'High_Vol': '#e67e22', 'Extreme_Vol': '#e74c3c'}

    # R² bars
    regimes = metrics_df['Regime'].tolist()
    r2_vals = metrics_df['R²'].tolist()
    bar_colors = [colors.get(r, 'gray') for r in regimes]

    axes[0].bar(regimes, r2_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("R² by Market Regime", fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

    # RMSE bars
    rmse_vals = metrics_df['RMSE'].tolist()
    axes[1].bar(regimes, rmse_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel("RMSE", fontsize=12)
    axes[1].set_title("RMSE by Market Regime", fontsize=13, fontweight='bold')
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + 0.0002, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)

    # Sample distribution pie
    sample_counts = metrics_df['Samples'].tolist()
    axes[2].pie(sample_counts, labels=regimes, colors=bar_colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
    axes[2].set_title("Regime Distribution (Validation Set)", fontsize=13, fontweight='bold')

    plt.suptitle("Regime-Aware TFT: Performance Across Market States", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "regime_performance.png", dpi=150)
    plt.close()
    print(f"Saved regime_performance.png")

    # ========== PLOT 2: Scatter by Regime ==========
    fig, axes = plt.subplots(1, len(regime_metrics), figsize=(6 * len(regime_metrics), 5))
    if len(regime_metrics) == 1:
        axes = [axes]

    for i, row in metrics_df.iterrows():
        regime = row['Regime']
        subset = pred_df[pred_df['regime'] == regime]
        ax = axes[i]

        ax.scatter(subset['y_true'], subset['y_pred'], alpha=0.05, s=2,
                   color=colors.get(regime, 'gray'))
        mn, mx = subset['y_true'].min(), subset['y_true'].max()
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{regime}\nR²={row['R²']:.3f}, N={int(row['Samples'])}",
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')

    plt.suptitle("Prediction Accuracy Across Market Regimes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "regime_scatter.png", dpi=150)
    plt.close()
    print(f"Saved regime_scatter.png")

    # ========== TABLE 2: Per-Stock × Per-Regime R² Matrix ==========
    print("\n" + "=" * 60)
    print("  Stock × Regime R² Matrix")
    print("=" * 60)

    matrix = {}
    for symbol in pred_df['symbol'].unique():
        row = {}
        for regime in regime_order:
            subset = pred_df[(pred_df['symbol'] == symbol) & (pred_df['regime'] == regime)]
            if len(subset) > 5:
                row[regime] = r2_score(subset['y_true'], subset['y_pred'])
            else:
                row[regime] = np.nan
        matrix[symbol] = row

    matrix_df = pd.DataFrame(matrix).T
    matrix_df = matrix_df[regime_order] if all(c in matrix_df.columns for c in regime_order) else matrix_df
    print(matrix_df.round(3).to_string())
    matrix_df.to_csv(project_root / "stock_regime_r2_matrix.csv")
    print(f"\nSaved stock_regime_r2_matrix.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_df.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, fontsize=11)
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index, fontsize=10)

    # Add text annotations
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            val = matrix_df.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='R²')
    ax.set_title("R² by Stock × Market Regime", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "stock_regime_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved stock_regime_heatmap.png")


if __name__ == "__main__":
    regime_analysis()

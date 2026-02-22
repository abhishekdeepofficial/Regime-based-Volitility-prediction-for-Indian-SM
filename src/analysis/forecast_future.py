"""
Future Volatility Forecasting
Predicts next-day realized volatility BEYOND the dataset using rolling forecasts.
Uses Model A (with autoregressive input) to forecast 5 trading days ahead.
Results saved to reports/figures/future_forecast/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta

PROJECT = Path("/Users/abhishekdeep/Documents/Data/MTech/Project-Sem 2/RResearch_Project")
sys.path.append(str(PROJECT))

from config.settings import PROCESSED_DATA_DIR

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl

# ── Config ──────────────────────────────────────────────────────────
CKPT_DIR     = PROJECT / "checkpoints" / "tft"
OUT_DIR      = PROJECT / "reports" / "figures" / "future_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FUTURE_DAYS = 5   # forecast 5 trading days ahead
ENCODER_LEN   = 75  # 1 trading day lookback
PRED_LEN      = 1   # single-step prediction

STOCKS = [
    'ADANIPORTS', 'AXISBANK', 'COALINDIA', 'HAL', 'HDFCBANK',
    'ICICIBANK', 'INFY', 'LT', 'RELIANCE', 'SBIN',
    'TATASTEEL', 'TCS', 'TITAN', 'WIPRO'
]

print("=" * 60)
print("  FUTURE VOLATILITY FORECASTING")
print("=" * 60)

# ── 1. Load Data ────────────────────────────────────────────────────
print("\nLoading data...")
data = pd.read_parquet(PROCESSED_DATA_DIR / "pooled_data_with_garch.parquet")

# Prep (same as tft_dataset.py)
dates = data['datetime'].sort_values().unique()
date_map = {d: i for i, d in enumerate(dates)}
data['time_idx_global'] = data['datetime'].map(date_map)
data['regime'] = data['regime'].astype(str).fillna('Unknown')
data['symbol'] = data['symbol'].astype(str)
data['log_target_RV_75bar'] = np.log1p(data['target_RV_75bar'])

feature_cols = [
    'log_return', 'RV_1h', 'RV_half_day', 'RV_1d', 'RV_1w',
    'volume_zscore', 'market_correlation',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'garch_volatility', 'Jump_1d', 'Jump_ratio',
    'Parkinson_1d', 'GK_1d', 'RS_1d', 'BV_1d', 'vwap_distance',
    'RV_1d_lag1', 'RV_1d_lag12', 'RV_1d_lag75', 'RV_1w_lag375', 'log_RV_1d',
]
for col in feature_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)

for col in ['is_opening', 'is_closing']:
    if col in data.columns:
        data[col] = data[col].astype(str)

data = data.dropna(subset=['log_target_RV_75bar'])
data = data.dropna()

print(f"  Rows: {len(data):,}")
print(f"  Stocks: {len(data['symbol'].unique())}")

# Get last date per stock
last_dates = data.groupby('symbol')['datetime'].max()
print(f"\n  Dataset end dates (per stock):")
for stock in sorted(STOCKS):
    if stock in last_dates.index:
        print(f"    {stock}: {last_dates[stock]}")

# ── 2. Find best checkpoint ────────────────────────────────────────
ckpts = sorted(CKPT_DIR.glob("tft-epoch=14-val_loss=0.0061*.ckpt"))
if not ckpts:
    # Fallback: find checkpoint with lowest val_loss
    import re
    all_ckpts = list(CKPT_DIR.glob("tft-epoch=*.ckpt"))
    best = None
    best_loss = float('inf')
    for c in all_ckpts:
        m = re.search(r'val_loss=([\d.]+)', c.name)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best = c
    ckpts = [best] if best else []

ckpt = ckpts[0]
print(f"\n  Loading: {ckpt.name}")

# ── 3. Build TimeSeriesDataSet for the full data ───────────────────
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
        time_varying_known_categoricals.append(col)

training_cutoff = data['time_idx_global'].max() - PRED_LEN - 1000

training_dataset = TimeSeriesDataSet(
    data[data['time_idx_global'] <= training_cutoff],
    time_idx="time_idx_global",
    target=target,
    group_ids=["symbol"],
    min_encoder_length=ENCODER_LEN // 2,
    max_encoder_length=ENCODER_LEN,
    min_prediction_length=1,
    max_prediction_length=PRED_LEN,
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

# ── 4. Load Model ──────────────────────────────────────────────────
tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
tft.eval()
print(f"  Model loaded: {sum(p.numel() for p in tft.parameters()):,} params")

# ── 5. Get last window for each stock and predict forward ──────────
# For each stock, take the last ENCODER_LEN bars => predict next day
# Then simulate rolling forward by using the prediction as new input

max_tidx = data['time_idx_global'].max()
results = {}

print(f"\n  Forecasting {N_FUTURE_DAYS} days ahead for {len(STOCKS)} stocks...")

for stock in STOCKS:
    stock_data = data[data['symbol'] == stock].sort_values('time_idx_global')
    
    if len(stock_data) < ENCODER_LEN + 1:
        print(f"  SKIP {stock}: insufficient data")
        continue
    
    stock_predictions = []
    last_actual_rv = np.expm1(stock_data['log_target_RV_75bar'].iloc[-1])
    last_date = pd.Timestamp(stock_data['datetime'].iloc[-1])
    
    # Get the last known values for rolling forward
    last_window = stock_data.tail(ENCODER_LEN + PRED_LEN).copy()
    
    for day in range(N_FUTURE_DAYS):
        # Create a prediction dataset from current window
        window = last_window.tail(ENCODER_LEN + PRED_LEN).copy()
        
        try:
            pred_ds = TimeSeriesDataSet.from_dataset(
                training_dataset,
                window,
                predict=True,
                stop_randomization=True,
            )
            pred_dl = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            # Predict
            with torch.no_grad():
                preds = tft.predict(pred_dl, mode="raw")
                pred_val = preds['prediction'][0, 0, 0].item()
            
            # Convert from log(1+RV) to RV
            pred_rv = np.expm1(pred_val)
            
            # Future date (skip weekends)
            future_date = last_date + timedelta(days=1)
            while future_date.weekday() >= 5:  # Skip Sat/Sun
                future_date += timedelta(days=1)
            
            stock_predictions.append({
                'day': day + 1,
                'date': future_date,
                'predicted_rv': pred_rv,
                'log_pred': pred_val,
            })
            
            # Roll forward: shift window, use prediction as new target
            new_row = last_window.iloc[-1:].copy()
            new_row['time_idx_global'] = new_row['time_idx_global'].values[0] + 1
            new_row['log_target_RV_75bar'] = pred_val
            new_row['datetime'] = future_date
            
            # Update RV features with prediction (approximate)
            new_row['RV_1d'] = pred_rv
            new_row['RV_1d_lag1'] = last_window['RV_1d'].iloc[-1]
            new_row['log_RV_1d'] = np.log1p(pred_rv)
            
            # Update temporal encodings
            hour = 9.25  # market open
            new_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            new_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            dow = future_date.weekday()
            new_row['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            new_row['dow_cos'] = np.cos(2 * np.pi * dow / 7)
            month = future_date.month
            new_row['month_sin'] = np.sin(2 * np.pi * month / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            last_window = pd.concat([last_window, new_row], ignore_index=True)
            last_date = future_date
            
        except Exception as e:
            print(f"    {stock} day {day+1} failed: {e}")
            # Use last prediction or last actual
            pred_rv = stock_predictions[-1]['predicted_rv'] if stock_predictions else last_actual_rv
            future_date = last_date + timedelta(days=1)
            while future_date.weekday() >= 5:
                future_date += timedelta(days=1)
            stock_predictions.append({
                'day': day + 1,
                'date': future_date,
                'predicted_rv': pred_rv,
                'log_pred': np.log1p(pred_rv),
            })
            last_date = future_date
    
    results[stock] = stock_predictions
    preds_str = ", ".join([f"Day{p['day']}={p['predicted_rv']:.4f}" for p in stock_predictions])
    print(f"  {stock}: {preds_str}")

# ── 6. Visualizations ──────────────────────────────────────────────
print(f"\nGenerating visualizations in {OUT_DIR}...")

# Set style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#444',
    'font.size': 10,
})

# ── 6a. Combined forecast plot (all stocks) ─────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Next-Day Volatility Forecast: 5 Trading Days Ahead',
             fontsize=18, fontweight='bold', color='#e94560', y=0.98)

for idx, stock in enumerate(sorted(STOCKS)):
    ax = axes[idx // 4, idx % 4]
    
    if stock not in results:
        ax.set_visible(False)
        continue
    
    stock_data = data[data['symbol'] == stock].sort_values('time_idx_global')
    
    # Last 10 actual points (historical)
    hist = stock_data.tail(10)
    hist_rv = np.expm1(hist['log_target_RV_75bar'].values)
    hist_dates = range(-len(hist_rv) + 1, 1)
    
    # Future predictions
    preds = results[stock]
    future_rv = [p['predicted_rv'] for p in preds]
    future_dates = range(1, len(future_rv) + 1)
    
    # Plot historical
    ax.plot(hist_dates, hist_rv, 'o-', color='#00d2ff', linewidth=2,
            markersize=4, label='Actual (Historical)', alpha=0.9)
    
    # Plot forecast
    ax.plot(future_dates, future_rv, 's--', color='#e94560', linewidth=2,
            markersize=6, label='Forecast', alpha=0.9)
    
    # Connect last actual to first forecast
    ax.plot([0, 1], [hist_rv[-1], future_rv[0]], '--', color='#888', linewidth=1)
    
    # Vertical line at today
    ax.axvline(x=0.5, color='#ffd700', linewidth=1.5, linestyle=':', alpha=0.7, label='Dataset End')
    
    # Shade forecast region
    ax.axvspan(0.5, len(future_rv) + 0.5, alpha=0.1, color='#e94560')
    
    ax.set_title(stock, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Days (relative)')
    ax.set_ylabel('RV (annualized)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)

# Hide extra subplots
for idx in range(len(STOCKS), 16):
    axes[idx // 4, idx % 4].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / 'all_stocks_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved all_stocks_forecast.png")

# ── 6b. Per-stock individual forecast plots ─────────────────────
for stock in sorted(STOCKS):
    if stock not in results:
        continue
    
    stock_data = data[data['symbol'] == stock].sort_values('time_idx_global')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Last 20 actual data points
    hist = stock_data.tail(20)
    hist_rv = np.expm1(hist['log_target_RV_75bar'].values)
    hist_dates = list(range(-len(hist_rv) + 1, 1))
    
    # Future
    preds = results[stock]
    future_rv = [p['predicted_rv'] for p in preds]
    future_dates = list(range(1, len(future_rv) + 1))
    
    # Plot
    ax.plot(hist_dates, hist_rv, 'o-', color='#00d2ff', linewidth=2.5,
            markersize=5, label='Actual (Historical)', zorder=3)
    ax.plot(future_dates, future_rv, 's-', color='#e94560', linewidth=2.5,
            markersize=8, label='Forecast (Future)', zorder=3)
    
    # Connect
    ax.plot([0, 1], [hist_rv[-1], future_rv[0]], '--', color='#888', linewidth=1.5)
    
    # Shade
    ax.axvline(x=0.5, color='#ffd700', linewidth=2, linestyle=':', alpha=0.8, label='Dataset End')
    ax.axvspan(0.5, max(future_dates) + 0.5, alpha=0.08, color='#e94560')
    
    # Annotate forecast values
    for p in preds:
        ax.annotate(f'{p["predicted_rv"]:.4f}',
                    xy=(p['day'], p['predicted_rv']),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=9, color='#e94560', fontweight='bold')
    
    ax.set_title(f'{stock} — Next-Day RV Forecast ({N_FUTURE_DAYS} Days Ahead)',
                 fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Trading Days (0 = last observed)', fontsize=11)
    ax.set_ylabel('Realized Volatility (annualized)', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'forecast_{stock}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Saved {len(STOCKS)} per-stock forecast plots")

# ── 6c. Summary bar chart: predicted next-day RV ────────────────
fig, ax = plt.subplots(figsize=(14, 6))

stocks_sorted = sorted(results.keys())
day1_rvs = [results[s][0]['predicted_rv'] for s in stocks_sorted]

colors = ['#e94560' if rv > np.median(day1_rvs) else '#00d2ff' for rv in day1_rvs]
bars = ax.bar(stocks_sorted, day1_rvs, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

# Add value labels
for bar, rv in zip(bars, day1_rvs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f'{rv:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')

ax.set_title('Predicted Next-Day Realized Volatility (Day 1 Forecast)',
             fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('Predicted RV (annualized)', fontsize=11)
ax.set_xlabel('Stock', fontsize=11)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.2)

# Add median line
median_rv = np.median(day1_rvs)
ax.axhline(y=median_rv, color='#ffd700', linewidth=1.5, linestyle='--',
           label=f'Median = {median_rv:.4f}', alpha=0.8)
ax.legend(fontsize=10)

plt.tight_layout()
fig.savefig(OUT_DIR / 'day1_forecast_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved day1_forecast_summary.png")

# ── 6d. Multi-day forecast heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Build matrix
matrix = []
for stock in sorted(results.keys()):
    row = [results[stock][d]['predicted_rv'] for d in range(N_FUTURE_DAYS)]
    matrix.append(row)

matrix = np.array(matrix)
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(N_FUTURE_DAYS))
ax.set_xticklabels([f'Day {d+1}' for d in range(N_FUTURE_DAYS)])
ax.set_yticks(range(len(stocks_sorted)))
ax.set_yticklabels(stocks_sorted)

# Add text annotations
for i in range(len(stocks_sorted)):
    for j in range(N_FUTURE_DAYS):
        ax.text(j, i, f'{matrix[i, j]:.4f}',
                ha='center', va='center', fontsize=8,
                color='black' if matrix[i, j] < matrix.max() * 0.7 else 'white')

plt.colorbar(im, ax=ax, label='Predicted RV')
ax.set_title('5-Day Ahead Volatility Forecast Heatmap',
             fontsize=14, fontweight='bold', color='white')

plt.tight_layout()
fig.savefig(OUT_DIR / 'forecast_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved forecast_heatmap.png")

# ── 7. Save CSV ─────────────────────────────────────────────────
rows = []
for stock in sorted(results.keys()):
    for p in results[stock]:
        rows.append({
            'stock': stock,
            'forecast_day': p['day'],
            'forecast_date': p['date'].strftime('%Y-%m-%d'),
            'predicted_rv': p['predicted_rv'],
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / 'future_forecasts.csv', index=False)
print(f"  Saved future_forecasts.csv")

print(f"\n{'=' * 60}")
print(f"  All outputs saved to: {OUT_DIR}")
print(f"{'=' * 60}")

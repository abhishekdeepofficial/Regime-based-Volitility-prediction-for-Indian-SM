"""
# Model B v2 Training — Kaggle GPU Notebook
# ==========================================
# 
# SETUP INSTRUCTIONS:
# 1. Go to kaggle.com → New Notebook
# 2. Settings → Accelerator → GPU T4 x2
# 3. Upload your data as a Dataset:
#    - Go to kaggle.com/datasets → New Dataset
#    - Upload: pooled_data_with_garch.parquet
#    - Name it: "volatility-data"
# 4. In your notebook → Add Data → search "volatility-data" → Add
# 5. Paste this entire file into a single cell and run
#
# The data will be at: /kaggle/input/volatility-data/pooled_data_with_garch.parquet
# Output saved to:     /kaggle/working/checkpoints/
"""

# ── Cell 1: Install dependencies ─────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "pytorch-forecasting", "lightning"], check=True)

# ── Cell 2: Imports ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE

import warnings
warnings.filterwarnings("ignore")

# ── GPU Check ────────────────────────────────────────────────────
print("=" * 60)
print("  GPU CHECK")
print("=" * 60)
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✅ GPU: {gpu} ({mem:.1f} GB)")
else:
    print("  ❌ No GPU! Go to Settings → Accelerator → GPU T4 x2")

# ── Cell 3: Load & Prep Data ────────────────────────────────────
print("\n" + "=" * 60)
print("  LOADING DATA")
print("=" * 60)

# Kaggle input path
DATA_PATH = Path("/kaggle/input/volatility-data/pooled_data_with_garch.parquet")
# Fallback: if uploaded differently
if not DATA_PATH.exists():
    candidates = list(Path("/kaggle/input").rglob("pooled_data_with_garch.parquet"))
    if candidates:
        DATA_PATH = candidates[0]
        print(f"  Found data at: {DATA_PATH}")
    else:
        raise FileNotFoundError(
            "Data not found! Upload pooled_data_with_garch.parquet as a Kaggle Dataset.\n"
            "Steps: kaggle.com/datasets → New Dataset → Upload the parquet file"
        )

data = pd.read_parquet(DATA_PATH)
print(f"  Loaded: {len(data):,} rows, {data['symbol'].nunique()} stocks")

# Create time index
dates = data['datetime'].sort_values().unique()
date_map = {d: i for i, d in enumerate(dates)}
data['time_idx_global'] = data['datetime'].map(date_map)

# Handle categoricals
data['regime'] = data['regime'].astype(str).fillna('Unknown')
data['symbol'] = data['symbol'].astype(str)

# Log target
data['log_target_RV_75bar'] = np.log1p(data['target_RV_75bar'])

# Fill NaNs in features
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

# Convert binary flags to strings for categorical encoding
for col in ['is_opening', 'is_closing']:
    if col in data.columns:
        data[col] = data[col].astype(str)

data = data.dropna(subset=['log_target_RV_75bar']).dropna()
print(f"  After cleanup: {len(data):,} rows")

# ── Cell 4: Create Dataset (NO autoregressive input) ────────────
print("\n" + "=" * 60)
print("  CREATING DATASET (Model B — No AR)")
print("=" * 60)

target = 'log_target_RV_75bar'
max_enc = 75
max_pred = 1

# Build feature lists — NO target in unknown reals (this is Model B)
time_varying_unknown_reals = [
    'log_return', 'RV_1h', 'RV_half_day', 'RV_1d', 'RV_1w',
    'volume_zscore', 'market_correlation',
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

print(f"  Unknown reals ({len(time_varying_unknown_reals)}): {time_varying_unknown_reals}")
print(f"  [Model B] Target NOT included as input ✅")

training_cutoff = data['time_idx_global'].max() - max_pred - 1000

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

batch_size = 128  # Larger batch on cloud GPU
train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=2)

print(f"  Train samples: {len(train_ds):,}")
print(f"  Val samples:   {len(val_ds):,}")

# ── Cell 5: Create Model ────────────────────────────────────────
print("\n" + "=" * 60)
print("  BUILDING MODEL B v2")
print("=" * 60)

tft = TemporalFusionTransformer.from_dataset(
    train_ds,
    learning_rate=0.0005,       # Slower for stable convergence
    hidden_size=256,            # 2x larger than v1
    attention_head_size=8,      # 2x more attention heads
    dropout=0.1,                # Less dropout
    hidden_continuous_size=128, # Better feature embeddings
    output_size=1,
    loss=RMSE(),
    log_interval=10,
    reduce_on_plateau_patience=10,
)

print(f"  Parameters: {sum(p.numel() for p in tft.parameters()):,}")

# ── Cell 6: Train ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING MODEL B v2")
print("=" * 60)

ckpt_dir = Path("/kaggle/working/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

trainer = pl.Trainer(
    max_epochs=150,
    accelerator="auto",
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.05,
    callbacks=[
        LearningRateMonitor(),
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=True, mode="min"),
        ModelCheckpoint(
            monitor="val_loss", dirpath=str(ckpt_dir),
            filename="model_b_v2-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3, mode="min"
        ),
    ],
    logger=TensorBoardLogger("/kaggle/working/logs", name="model_b_v2"),
    limit_train_batches=1500,
    limit_val_batches=300,
)

trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

best_ckpt = trainer.checkpoint_callback.best_model_path
best_loss = trainer.checkpoint_callback.best_model_score
print(f"\n  ✅ Best checkpoint: {best_ckpt}")
print(f"  ✅ Best val_loss:   {best_loss:.6f}")

# ── Cell 7: Evaluate ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATING MODEL B v2")
print("=" * 60)

tft_best = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
outputs = tft_best.predict(val_dl, return_y=True)

y_pred = outputs.output.cpu().numpy().flatten()
y_true = outputs.y[0].cpu().numpy().flatten()

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  Model B v1 → v2 Comparison             │")
print(f"  ├──────────┬──────────┬──────────┬─────────┤")
print(f"  │ Metric   │  v1      │  v2      │ Change  │")
print(f"  ├──────────┼──────────┼──────────┼─────────┤")
print(f"  │ R²       │  0.2576  │  {r2:.4f}  │ {r2-0.2576:+.4f} │")
print(f"  │ RMSE     │  0.0521  │  {rmse:.4f}  │ {rmse-0.0521:+.4f} │")
print(f"  │ MAPE     │  19.26%  │  {mape:.2f}%  │ {mape-19.26:+.2f}pp │")
print(f"  └──────────┴──────────┴──────────┴─────────┘")

print(f"\n  📥 Download checkpoint from: {best_ckpt}")
print(f"  📥 Copy to local: checkpoints/tft_model_b/")
print(f"\n{'=' * 60}")
print(f"  DONE! Download the .ckpt file to your local project.")
print(f"{'=' * 60}")

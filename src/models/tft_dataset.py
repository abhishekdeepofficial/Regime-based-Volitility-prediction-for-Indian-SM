import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.settings import PROCESSED_DATA_DIR

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except ImportError:
    print("pytorch_forecasting not installed.")

def load_and_prep_tft_data() -> Tuple[pd.DataFrame, int]:
    """
    Load pooled data and prepare for TFT TimeSeriesDataSet.
    Uses pooled_data_with_garch.parquet (has regimes + GARCH + all features).
    Falls back to pooled_data_with_regimes.parquet if GARCH not available.
    """
    # Try GARCH-enriched data first (has all features)
    garch_path = PROCESSED_DATA_DIR / "pooled_data_with_garch.parquet"
    regime_path = PROCESSED_DATA_DIR / "pooled_data_with_regimes.parquet"
    
    if garch_path.exists():
        path = garch_path
        print(f"Loading {garch_path.name} (with GARCH features)...")
    elif regime_path.exists():
        path = regime_path
        print(f"Loading {regime_path.name} (GARCH features missing, using fallback)...")
    else:
        raise FileNotFoundError(
            f"Neither {garch_path} nor {regime_path} found. Run the pipeline first."
        )
    
    data = pd.read_parquet(path)
    
    # 1. Create Integer Time Index (Global)
    dates = data['datetime'].sort_values().unique()
    date_map = {d: i for i, d in enumerate(dates)}
    data['time_idx_global'] = data['datetime'].map(date_map)
    
    # 2. Handle Categoricals
    data['regime'] = data['regime'].astype(str).fillna('Unknown')
    data['symbol'] = data['symbol'].astype(str)
    
    # 3. Create log-transformed target (handles skewed distribution, keeps values manageable)
    target_col = 'target_RV_75bar'
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    
    data['log_target_RV_75bar'] = np.log1p(data[target_col])
    
    # 4. Drop NaN rows
    data = data.dropna(subset=['log_target_RV_75bar'])
    
    # Fill NaNs in feature columns with 0 (some features may have edge NaNs)
    feature_cols = [
        'log_return', 'RV_1h', 'RV_half_day', 'RV_1d', 'RV_1w',
        'volume_zscore', 'market_correlation',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    ]
    
    # Optional features (may not exist if GARCH step wasn't run)
    optional_features = [
        'garch_volatility', 'Jump_1d', 'Jump_ratio',
        'Parkinson_1d', 'GK_1d', 'RS_1d',
        'BV_1d', 'circuit_breaker', 'Jump_sig',
        'is_opening', 'is_closing',
    ]
    
    for col in feature_cols + optional_features:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    initial_len = len(data)
    data = data.dropna()
    print(f"Dropped {initial_len - len(data)} rows with NaNs. Final rows: {len(data)}")
    
    return data

def create_tft_dataset(data: pd.DataFrame, max_encoder_length=75, max_prediction_length=1, use_autoregressive=True):
    """
    Create TimeSeriesDataSet with all available features.
    
    Args:
        max_encoder_length: Lookback period (75 bars = 1 day)
        max_prediction_length: Prediction horizon (1 step = predict next-day vol)
        use_autoregressive: If True, include target as input feature (Model A).
                            If False, exclude it (Model B — ablation study).
    """
    # Use log-transformed target
    target = 'log_target_RV_75bar'
    
    # Build feature lists dynamically based on available columns
    time_varying_unknown_reals = [
        'log_return', 
        'RV_1h', 'RV_half_day', 'RV_1d', 'RV_1w',
        'volume_zscore', 'market_correlation',
    ]
    
    # Autoregressive input — controlled by flag for ablation study
    if use_autoregressive:
        time_varying_unknown_reals.append(target)
        print("  [Model A] Autoregressive target included as input")
    
    # Add optional features if present
    optional_unknown_reals = [
        'garch_volatility', 'Jump_1d', 'Jump_ratio',
        'Parkinson_1d', 'GK_1d', 'RS_1d',
        'BV_1d', 'vwap_distance',
        # HAR-RV lagged features
        'RV_1d_lag1', 'RV_1d_lag12', 'RV_1d_lag75',
        'RV_1w_lag375', 'log_RV_1d',
    ]
    for col in optional_unknown_reals:
        if col in data.columns:
            time_varying_unknown_reals.append(col)
    
    # Known future features (time-based, deterministic)
    time_varying_known_reals = ['time_idx_global']
    optional_known_reals = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    for col in optional_known_reals:
        if col in data.columns:
            time_varying_known_reals.append(col)
    
    # Categoricals
    time_varying_unknown_categoricals = ['regime']
    
    time_varying_known_categoricals = []
    optional_known_cats = ['is_opening', 'is_closing']
    for col in optional_known_cats:
        if col in data.columns:
            # Ensure they are strings for categorical handling
            data[col] = data[col].astype(str)
            time_varying_known_categoricals.append(col)
    
    print(f"TFT Features:")
    print(f"  Unknown reals ({len(time_varying_unknown_reals)}): {time_varying_unknown_reals}")
    print(f"  Known reals ({len(time_varying_known_reals)}): {time_varying_known_reals}")
    print(f"  Known cats ({len(time_varying_known_categoricals)}): {time_varying_known_categoricals}")
    
    # Training cutoff
    training_cutoff = data['time_idx_global'].max() - max_prediction_length - 1000
    
    training_dataset = TimeSeriesDataSet(
        data[data['time_idx_global'] <= training_cutoff],
        time_idx="time_idx_global",
        target=target,
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # Static Inputs
        static_categoricals=["symbol"],
        
        # Known inputs (time features)
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        
        # Observed Inputs (past only)
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        
        # Normalize target per-group (softplus keeps positive)
        target_normalizer=GroupNormalizer(
            groups=["symbol"], transformation="softplus"
        ),
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    # Validation Set
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, 
        data, 
        predict=False, 
        stop_randomization=True,
        min_prediction_idx=training_cutoff + 1,
        min_prediction_length=max_prediction_length, 
    )
    
    return training_dataset, validation_dataset

if __name__ == "__main__":
    df = load_and_prep_tft_data()
    train_ds, val_ds = create_tft_dataset(df)
    print("Dataset created successfully.")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Check loaders
    batch_size = 64
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    print("Dataloader check passed.")

#!/usr/bin/env python3
"""
Lightning.ai Pipeline: Train Model B v2 on Cloud GPU
Run this after uploading the project and running setup_lightning.sh

Usage:
    python run_pipeline.py              # Full pipeline (data prep + train)
    python run_pipeline.py --train-only # Skip data prep, just train
"""
import sys
import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT))


def step_1_verify_gpu():
    """Verify CUDA GPU is available."""
    import torch
    print("=" * 60)
    print("  STEP 1: GPU Verification")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  ✅ GPU: {gpu} ({mem:.1f} GB)")
        return True
    else:
        print("  ❌ No CUDA GPU found! Training will be very slow on CPU.")
        print("  → Make sure you selected a GPU machine in Lightning.ai")
        return False


def step_2_data_prep():
    """Run feature engineering pipeline if data not ready."""
    processed = PROJECT / "data" / "processed" / "pooled_data_with_garch.parquet"
    
    print("\n" + "=" * 60)
    print("  STEP 2: Data Preparation")
    print("=" * 60)
    
    if processed.exists():
        import pandas as pd
        df = pd.read_parquet(processed)
        print(f"  ✅ Data already exists: {len(df):,} rows, {len(df['symbol'].unique())} stocks")
        return True
    else:
        print("  ❌ Processed data not found!")
        print(f"  → Expected: {processed}")
        print("  → Upload your data/processed/pooled_data_with_garch.parquet")
        return False


def step_3_train_model_b():
    """Train improved Model B."""
    print("\n" + "=" * 60)
    print("  STEP 3: Training Model B v2 (Improved)")
    print("=" * 60)
    
    from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    
    # Load data
    data = load_and_prep_tft_data()
    
    max_enc = 75
    max_pred = 1
    
    # Model B: NO autoregressive input
    train_ds, val_ds = create_tft_dataset(data, max_enc, max_pred, use_autoregressive=False)
    
    batch_size = 128  # Larger batch on cloud GPU
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=4)
    
    # Improved hyperparameters for Model B
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=0.0005,
        hidden_size=256,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=128,
        output_size=1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=10,
    )
    print(f"  Parameters: {tft.size()/1e3:.1f}k")
    
    ckpt_dir = PROJECT / "checkpoints" / "tft_model_b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=20,
        verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(
        monitor="val_loss", dirpath=str(ckpt_dir),
        filename="model_b_v2-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs", name="model_b_v2")
    
    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="auto",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.05,
        callbacks=[lr_logger, early_stop, checkpoint],
        logger=logger,
        limit_train_batches=1500,
        limit_val_batches=300,
    )
    
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    print(f"\n  ✅ Best checkpoint: {checkpoint.best_model_path}")
    print(f"  ✅ Best val_loss: {checkpoint.best_model_score:.6f}")
    return checkpoint.best_model_path


def step_4_evaluate():
    """Evaluate the new Model B checkpoint."""
    print("\n" + "=" * 60)
    print("  STEP 4: Evaluation")
    print("=" * 60)
    
    ckpt_dir = PROJECT / "checkpoints" / "tft_model_b"
    ckpts = sorted(ckpt_dir.glob("model_b_v2*.ckpt"), key=lambda x: x.stat().st_mtime)
    
    if not ckpts:
        print("  ❌ No v2 checkpoint found")
        return
    
    best = ckpts[-1]
    print(f"  Loading: {best.name}")
    
    from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
    from pytorch_forecasting import TemporalFusionTransformer
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    
    data = load_and_prep_tft_data()
    _, val_ds = create_tft_dataset(data, use_autoregressive=False)
    val_dl = val_ds.to_dataloader(train=False, batch_size=128, num_workers=4)
    
    tft = TemporalFusionTransformer.load_from_checkpoint(str(best))
    outputs = tft.predict(val_dl, return_y=True)
    
    y_pred = outputs.output.cpu().numpy().flatten()
    y_true = outputs.y[0].cpu().numpy().flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n  Model B v2 Results:")
    print(f"    R²   = {r2:.4f}  (v1 was 0.2576)")
    print(f"    RMSE = {rmse:.6f}  (v1 was 0.0521)")
    print(f"    MAPE = {mape:.2f}%  (v1 was 19.26%)")
    print(f"\n  Improvement: R² {0.2576:.3f} → {r2:.3f} ({r2 - 0.2576:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true", help="Skip data prep, just train")
    args = parser.parse_args()
    
    has_gpu = step_1_verify_gpu()
    
    if not args.train_only:
        has_data = step_2_data_prep()
        if not has_data:
            sys.exit(1)
    
    best_path = step_3_train_model_b()
    step_4_evaluate()
    
    print("\n" + "=" * 60)
    print("  DONE! Download checkpoints/tft_model_b/ to your local project")
    print("=" * 60)

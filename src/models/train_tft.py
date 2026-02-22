import pandas as pd
import json
import sys
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, RMSE

def load_best_params():
    """Load tuned hyperparameters from config/best_params.json if available."""
    params_path = project_root / "config" / "best_params.json"
    defaults = {
        "learning_rate": 0.001,
        "hidden_size": 128,
        "dropout": 0.15,
        "hidden_continuous_size": 64,
        "attention_head_size": 4,
    }
    if params_path.exists():
        try:
            with open(params_path) as f:
                params = json.load(f)
            print(f"Loaded tuned params: {params}")
            return {**defaults, **params}
        except Exception as e:
            print(f"Failed to load best_params.json: {e}, using defaults")
    return defaults

def train_tft():
    print("Starting TFT Training Pipeline...")
    
    # 1. Data
    data = load_and_prep_tft_data()
    
    max_enc = 75   # 1 day of lookback (safe for memory)
    max_pred = 1   # Single-step: predict next-day RV (75 bars ahead)
    
    train_ds, val_ds = create_tft_dataset(data, max_enc, max_pred)
    
    batch_size = 64
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    
    # 2. Model — use tuned hyperparameters
    params = load_best_params()
    
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        hidden_continuous_size=params["hidden_continuous_size"],
        output_size=1,
        loss=RMSE(),
        log_interval=10, 
        reduce_on_plateau_patience=6,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    # 3. Trainer — train on FULL data, more epochs
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=15, 
        verbose=True, 
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(project_root / "checkpoints" / "tft"),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        limit_train_batches=1000,   # 1000 batches × 64 = 64k samples/epoch
        limit_val_batches=200,
    )
    
    # 4. Fit
    print("Fitting model on FULL dataset...")
    trainer.fit(
        tft,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )
    
    # 5. Best Model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train_tft()

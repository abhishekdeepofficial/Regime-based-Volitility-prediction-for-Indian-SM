"""
Model B: TFT without autoregressive input (ablation study).
This trains the same model architecture but WITHOUT the target
as an input feature, forcing the model to predict from
backward-looking features only.
"""
import sys
from pathlib import Path
import json
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE


def load_best_params():
    """Model B needs different hyperparameters than Model A.
    Without the dominant AR signal, it needs larger capacity and more patience."""
    return {
        "learning_rate": 0.0005,       # Slower LR for stable convergence on weak signals
        "hidden_size": 256,            # More capacity to learn complex non-AR patterns
        "dropout": 0.1,                # Less regularization — model needs to fit harder
        "hidden_continuous_size": 128,  # Better numerical feature embeddings
        "attention_head_size": 8,       # More attention heads for diverse temporal patterns
    }


def train_model_b():
    print("=" * 50)
    print("  MODEL B: TFT WITHOUT Autoregressive Input")
    print("  (Ablation Study — genuine predictive power)")
    print("=" * 50)

    data = load_and_prep_tft_data()

    max_enc = 75    # 1 day of lookback (keep same as Model A for fair comparison)
    max_pred = 1

    # KEY DIFFERENCE: use_autoregressive=False
    train_ds, val_ds = create_tft_dataset(data, max_enc, max_pred, use_autoregressive=False)

    batch_size = 64
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

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
        reduce_on_plateau_patience=10,
    )
    print(f"Parameters: {tft.size()/1e3:.1f}k")

    # Save to separate directory
    ckpt_dir = project_root / "checkpoints" / "tft_model_b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=str(ckpt_dir),
                                 filename="model_b-{epoch:02d}-{val_loss:.4f}", save_top_k=3, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs", name="model_b")

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

    print(f"\nModel B best checkpoint: {checkpoint.best_model_path}")
    print(f"Model B best val_loss: {checkpoint.best_model_score:.6f}")


if __name__ == "__main__":
    train_model_b()

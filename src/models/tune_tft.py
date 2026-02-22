import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset

def objective(trial):
    # 1. Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32])
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    
    print(f"Trial Params: Learing Rate={learning_rate}, Hidden={hidden_size}, Dropout={dropout}")

    # 2. Data
    # Load once outside if possible? But here we need fresh loaders maybe? 
    # Or reuse global data.
    # For efficiency, we should load data outside objective, but okay for now.
    
    # Use global variable for data to avoid reloading every trial
    global data_global
    
    max_enc = 75
    max_pred = 12
    batch_size = 128
    
    try:
        train_ds, val_ds = create_tft_dataset(data_global, max_enc, max_pred)
        train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
        
        # 3. Model
        tft = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
        
        # 4. Trainer
        # Fast training for tuning
        trainer = pl.Trainer(
            max_epochs=3, # Short epochs for tuning
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            limit_train_batches=50, # Very fast check
            limit_val_batches=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False # Disable logging for speed/cleanliness
        )
        
        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        
        # 5. Metric
        val_loss = trainer.callback_metrics["val_loss"].item()
        
        return val_loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

if __name__ == "__main__":
    print("Loading data for tuning...")
    data_global = load_and_prep_tft_data()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) # 5 trials for demo
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    # Save best params
    import json
    with open(project_root / "config" / "best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
        
    print("Saved best params to config/best_params.json")

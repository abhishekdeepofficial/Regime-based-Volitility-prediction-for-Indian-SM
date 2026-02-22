import pandas as pd
import numpy as np
import sys
from pathlib import Path
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.tft_dataset import load_and_prep_tft_data, create_tft_dataset
from src.evaluation.metrics import evaluate_forecasts, directional_accuracy

def find_best_checkpoint():
    """Find the best checkpoint from training."""
    # 1. Check dedicated checkpoint dir first
    ckpt_dir = project_root / "checkpoints" / "tft"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # Pick by best val_loss in filename, or most recent
            return sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    
    # 2. Search lightning_logs recursively
    log_dir = project_root / "lightning_logs"
    if log_dir.exists():
        ckpts = list(log_dir.rglob("*.ckpt"))
        if ckpts:
            return sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    
    # 3. Search project root
    ckpts = list(project_root.rglob("*.ckpt"))
    if ckpts:
        return sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    
    return None

def evaluate_model():
    print("Starting TFT Evaluation...")
    
    # 1. Load Data & Validation Set
    data = load_and_prep_tft_data()
    _, val_ds = create_tft_dataset(data)
    
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # 2. Load Best Model
    best_model_path = find_best_checkpoint()
    
    if best_model_path is None:
        print("No checkpoint found.")
        return
    
    print(f"Loading model from {best_model_path}")
    tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    # 3. Predict
    print("Generating predictions...")
    outputs = tft.predict(val_dl, return_y=True)
    y_pred = outputs.output 
    y_true = outputs.y[0] if isinstance(outputs.y, (list, tuple)) else outputs.y
        
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    
    print("\n--- Evaluation Results (Daily RV, log-transformed space) ---")
    print(f"Prediction: Next-day Realized Volatility (75-bar forward RV)")
    print(f"Samples: {len(y_pred_np)}")
    
    metrics = evaluate_forecasts(y_true_np, y_pred_np)
    da = directional_accuracy(y_true_np, y_pred_np)
    metrics['DA'] = da
    
    # MAPE
    mape = np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-8))) * 100
    metrics['MAPE'] = mape
    
    print(pd.Series(metrics))
    
    # Save results
    res_df = pd.DataFrame({'daily_RV': metrics}, index=metrics.keys())
    res_df.to_csv(project_root / "tft_evaluation_results.csv")
    print("\nSaved evaluation results.")

if __name__ == "__main__":
    evaluate_model()

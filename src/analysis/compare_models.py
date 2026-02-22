"""
Compare Model A (with autoregressive) vs Model B (without) for the ablation study.
Generates a comparison table and side-by-side plots.
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
from src.evaluation.metrics import evaluate_forecasts, directional_accuracy


def evaluate_single_model(model_path, data, use_autoregressive, label):
    """Evaluate one model and return metrics + predictions."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {label}")
    print(f"  Checkpoint: {model_path.name}")
    print(f"{'='*50}")

    _, val_ds = create_tft_dataset(data, use_autoregressive=use_autoregressive)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.load_from_checkpoint(str(model_path))
    outputs = tft.predict(val_dl, return_y=True)

    y_pred = outputs.output.cpu().numpy().flatten()
    y_true = outputs.y[0].cpu().numpy().flatten() if isinstance(outputs.y, (list, tuple)) else outputs.y.cpu().numpy().flatten()

    metrics = evaluate_forecasts(y_true, y_pred)
    metrics['DA'] = directional_accuracy(y_true, y_pred)
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  DA:   {metrics['DA']:.4f}")

    return metrics, y_pred, y_true


def find_best_ckpt(ckpt_dir):
    """Find best checkpoint by val_loss in filename."""
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def compare_models():
    print("Loading data...")
    data = load_and_prep_tft_data()

    # Find checkpoints
    model_a_ckpt = find_best_ckpt(project_root / "checkpoints" / "tft")
    model_b_ckpt = find_best_ckpt(project_root / "checkpoints" / "tft_model_b")

    if not model_a_ckpt:
        print("Model A checkpoint not found. Train it first.")
        return
    if not model_b_ckpt:
        print("Model B checkpoint not found. Run: python3 src/models/train_model_b.py")
        return

    # Evaluate both
    metrics_a, pred_a, true_a = evaluate_single_model(
        model_a_ckpt, data, use_autoregressive=True, label="Model A (with autoregressive)")
    metrics_b, pred_b, true_b = evaluate_single_model(
        model_b_ckpt, data, use_autoregressive=False, label="Model B (without autoregressive)")

    # Comparison Table
    print("\n" + "=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("=" * 60)
    comparison = pd.DataFrame({
        'Model A\n(with AR)': metrics_a,
        'Model B\n(without AR)': metrics_b,
    })
    print(comparison.to_string())

    # Save comparison
    comparison.to_csv(project_root / "ablation_comparison.csv")
    print(f"\nSaved to ablation_comparison.csv")

    # Comparison Plot
    report_dir = project_root / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Model A scatter
    axes[0].scatter(true_a, pred_a, alpha=0.05, s=2, color='steelblue')
    mn, mx = true_a.min(), true_a.max()
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
    axes[0].set_title(f"Model A: With AR Input (R²={metrics_a['R2']:.3f})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Actual Daily RV")
    axes[0].set_ylabel("Predicted Daily RV")

    # Model B scatter
    axes[1].scatter(true_b, pred_b, alpha=0.05, s=2, color='darkorange')
    mn, mx = true_b.min(), true_b.max()
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2)
    axes[1].set_title(f"Model B: Without AR Input (R²={metrics_b['R2']:.3f})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Actual Daily RV")
    axes[1].set_ylabel("Predicted Daily RV")

    plt.suptitle("Ablation Study: Effect of Autoregressive Input on TFT Volatility Forecasting",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "ablation_comparison.png", dpi=150)
    plt.close()
    print(f"Saved comparison plot to {report_dir / 'ablation_comparison.png'}")


if __name__ == "__main__":
    compare_models()

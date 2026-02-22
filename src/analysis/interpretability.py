"""
TFT Interpretability Analysis
Extracts and visualizes variable importance and attention weights from the TFT model.
Properly unpacks batch tuple (x, y) before passing to model.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
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


def interpretability_analysis():
    print("=" * 60)
    print("  TFT INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Load data
    data = load_and_prep_tft_data()
    _, val_ds = create_tft_dataset(data)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    best_ckpt = find_best_checkpoint()
    if not best_ckpt:
        print("No checkpoint found.")
        return
    print(f"Loading model: {best_ckpt.name}")
    tft = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))

    report_dir = project_root / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ========== Step 1: Get raw model output ==========
    print("\nRunning predictions with raw output mode...")
    tft.eval()
    tft.to('cpu')  # Move to CPU to avoid MPS placeholder storage issue

    def to_cpu(obj):
        """Recursively move batch data to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(to_cpu(v) for v in obj)
        return obj

    all_attention = []
    all_encoder_weights = []
    all_decoder_weights = []
    all_static_weights = []

    batch_count = 0
    max_batches = 50

    with torch.no_grad():
        for batch in val_dl:
            if batch_count >= max_batches:
                break
            try:
                # CRITICAL: batch is (x, y) tuple — pass x dict to forward
                x, y = batch
                x = to_cpu(x)
                out = tft(x)  # Pass in x dict, not the whole tuple
                interpretation = tft.interpret_output(out, reduction="mean")

                if 'attention' in interpretation:
                    all_attention.append(interpretation['attention'].cpu())
                if 'encoder_variables' in interpretation:
                    all_encoder_weights.append(interpretation['encoder_variables'].cpu())
                if 'decoder_variables' in interpretation:
                    all_decoder_weights.append(interpretation['decoder_variables'].cpu())
                if 'static_variables' in interpretation:
                    all_static_weights.append(interpretation['static_variables'].cpu())

                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processed {batch_count}/{max_batches} batches...")
            except Exception as e:
                print(f"  Batch {batch_count} error: {e}")
                # Try alternate approach: full forward with logging
                try:
                    x, y = batch
                    x = to_cpu(x)
                    out_raw = tft.forward(x)
                    print(f"  Raw output type: {type(out_raw)}")
                    if isinstance(out_raw, dict):
                        print(f"  Keys: {list(out_raw.keys())}")
                    elif isinstance(out_raw, tuple):
                        print(f"  Tuple len: {len(out_raw)}")
                    break
                except Exception as e2:
                    print(f"  Alternate approach also failed: {e2}")
                    break

    # ========== Step 2: Variable Importance ==========
    if all_encoder_weights:
        print("\n\nComputing Variable Importance...")
        enc_importance = torch.stack(all_encoder_weights).mean(dim=0).numpy()

        all_enc_vars = (
            val_ds.time_varying_unknown_reals +
            val_ds.time_varying_known_reals +
            val_ds.time_varying_unknown_categoricals +
            val_ds.time_varying_known_categoricals
        )

        if len(all_enc_vars) != len(enc_importance):
            print(f"  Note: {len(all_enc_vars)} names vs {len(enc_importance)} weights, using available")
            all_enc_vars = all_enc_vars[:len(enc_importance)]
            if len(all_enc_vars) < len(enc_importance):
                all_enc_vars.extend([f"var_{i}" for i in range(len(all_enc_vars), len(enc_importance))])

        importance_df = pd.DataFrame({
            'Variable': all_enc_vars,
            'Importance': enc_importance
        }).sort_values('Importance', ascending=False)

        print("\nEncoder Variable Importance:")
        print(importance_df.to_string(index=False))
        importance_df.to_csv(project_root / "variable_importance.csv", index=False)

        # --- Bar chart ---
        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(15, len(importance_df))
        top_df = importance_df.head(top_n).sort_values('Importance')

        colors = []
        for var in top_df['Variable']:
            if 'RV' in var or 'target' in var:
                colors.append('#e74c3c')
            elif 'garch' in var.lower():
                colors.append('#9b59b6')
            elif 'Jump' in var or 'BV' in var:
                colors.append('#e67e22')
            elif 'sin' in var or 'cos' in var or 'time' in var:
                colors.append('#3498db')
            elif 'regime' in var.lower():
                colors.append('#2ecc71')
            else:
                colors.append('#95a5a6')

        ax.barh(range(top_n), top_df['Importance'].values, color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_df['Variable'].values, fontsize=10)
        ax.set_xlabel("Importance Weight", fontsize=12)
        ax.set_title("TFT Encoder Variable Importance (Top 15)", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor='#e74c3c', label='RV / Target'),
            Patch(facecolor='#9b59b6', label='GARCH'),
            Patch(facecolor='#e67e22', label='Jump / BV'),
            Patch(facecolor='#3498db', label='Temporal'),
            Patch(facecolor='#2ecc71', label='Regime'),
            Patch(facecolor='#95a5a6', label='Other'),
        ]
        ax.legend(handles=legend_items, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(report_dir / "variable_importance.png", dpi=150)
        plt.close()
        print("  Saved variable_importance.png")
    else:
        print("\n  No encoder variable weights extracted.")

    # ========== Step 3: Static Variable Importance ==========
    if all_static_weights:
        print("\nStatic Variable Importance:")
        static_importance = torch.stack(all_static_weights).mean(dim=0).numpy()
        static_vars = list(val_ds.static_categoricals or []) + list(val_ds.static_reals or [])
        for i, imp in enumerate(static_importance):
            name = static_vars[i] if i < len(static_vars) else f"static_{i}"
            print(f"  {name}: {imp:.4f}")

    # ========== Step 4: Attention Weights ==========
    if all_attention:
        print("\nComputing Attention Weights...")
        avg_attention = torch.stack(all_attention).mean(dim=0).numpy()

        fig, ax = plt.subplots(figsize=(14, 5))
        positions = np.arange(len(avg_attention))

        ax.bar(positions, avg_attention, color='steelblue', alpha=0.8,
               edgecolor='black', linewidth=0.3)
        ax.set_xlabel("Encoder Position (bars before prediction)", fontsize=12)
        ax.set_ylabel("Attention Weight", fontsize=12)
        ax.set_title("TFT Self-Attention: Which Past Time Steps Matter Most",
                      fontsize=14, fontweight='bold')

        max_pos = np.argmax(avg_attention)
        bars_ago = len(avg_attention) - 1 - max_pos
        ax.annotate(f'Peak: {bars_ago} bars ago\n({bars_ago*5} min)',
                    xy=(max_pos, avg_attention[max_pos]),
                    xytext=(max_pos + 5, avg_attention[max_pos] * 1.15),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')

        n = len(avg_attention)
        if n > 12:
            ax.axvline(x=n-12, color='orange', linestyle='--', alpha=0.5, label='1 hour ago')
        if n > 1:
            ax.axvline(x=n-1, color='green', linestyle='--', alpha=0.5, label='Most recent bar')

        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "attention_weights.png", dpi=150)
        plt.close()
        print("  Saved attention_weights.png")
    else:
        print("\n  No attention weights extracted.")

    # ========== Step 5: Decoder Variable Importance ==========
    if all_decoder_weights:
        print("\nDecoder Variable Importance:")
        dec_importance = torch.stack(all_decoder_weights).mean(dim=0).numpy()
        dec_vars = (
            val_ds.time_varying_known_reals +
            val_ds.time_varying_known_categoricals
        )
        for i, imp in enumerate(dec_importance):
            name = dec_vars[i] if i < len(dec_vars) else f"decoder_{i}"
            print(f"  {name}: {imp:.4f}")

    print(f"\nModel summary:")
    print(f"  Parameters: {sum(p.numel() for p in tft.parameters()):,}")
    print(f"  Hidden size: {tft.hparams.hidden_size}")
    print(f"  Attention heads: {tft.hparams.attention_head_size}")
    print(f"\nAll interpretability outputs saved to {report_dir}")


if __name__ == "__main__":
    interpretability_analysis()

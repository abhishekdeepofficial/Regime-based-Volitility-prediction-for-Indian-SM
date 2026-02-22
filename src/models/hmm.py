import pandas as pd
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.settings import PROCESSED_DATA_DIR
from src.models.hmm_module import RegimeDetector

def run_regime_detection():
    print("Starting Regime Detection (Refactored)...")
    
    pooled_path = PROCESSED_DATA_DIR / "pooled_data.parquet"
    if not pooled_path.exists():
        print("Pooled data not found.")
        return
        
    print("Loading Pooled Data...")
    pooled_df = pd.read_parquet(pooled_path)
    
    print("Computing Market Average proxy...")
    market_df = pooled_df.groupby('datetime')[['log_return', 'RV_1d']].mean()
    
    detector = RegimeDetector(n_components=4)
    detector.fit(market_df)
    
    # Predict
    market_regimes = detector.predict(market_df)
    
    # Save Model
    model_path = PROCESSED_DATA_DIR / "hmm_market_model.pkl"
    joblib.dump(detector, model_path)
    print(f"Saved model to {model_path}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    subset = market_regimes.iloc[-1000:]
    sns.scatterplot(data=subset, x='datetime', y='log_return', hue='regime', palette='viridis', s=10)
    plt.title("Market Regimes (Last 1000 bars)")
    plt.savefig(PROCESSED_DATA_DIR / "regime_plot.png")
    print("Saved regime plot.")

if __name__ == "__main__":
    run_regime_detection()

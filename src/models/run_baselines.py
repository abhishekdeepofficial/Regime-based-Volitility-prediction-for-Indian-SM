import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config.settings import PROCESSED_DATA_DIR, STOCKS
from src.models.garch import train_garch_benchmarks
from src.models.har import HARRVModel

def load_processed_data(symbol: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{symbol}_processed.parquet"
    return pd.read_parquet(path)

def run_baselines():
    print("Starting Baseline Modeling...")
    
    results = []
    
    for symbol in tqdm(STOCKS):
        try:
            print(f"\nProcessing {symbol}...")
            df = load_processed_data(symbol)
            
            # 1. GARCH Models
            # Use deseasonalized returns (drop NaNs for GARCH)
            # Ensure we are using the 5-min returns for 5-min forecasting, but typically GARCH is daily.
            # The processed data has 'deseasonalized_return' at 5-min level.
            # To fit a daily GARCH, we should aggregate or fit on high-freq? 
            # The Prompt says: "Multi-scale GARCH", "Fit GARCH family to stock returns".
            # Usually GARCH on 5-min data is noisy. But let's follow the implied instruction to use the available returns.
            # Scaling is handled inside the GARCHModel wrapper (*100).
            
            returns = df['deseasonalized_return'].dropna()
            
            # Check for infinites
            if np.isinf(returns).any():
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            print(f"  Fitting GARCH models on {len(returns)} samples...")
            garch_results = train_garch_benchmarks(returns)
            
            # Select best GARCH by AIC
            if garch_results:
                best_garch = min(garch_results, key=lambda x: garch_results[x]['aic'])
                print(f"  Best GARCH: {best_garch} (AIC: {garch_results[best_garch]['aic']:.2f})")
                garch_aic = garch_results[best_garch]['aic']
            else:
                best_garch = "None"
                garch_aic = np.nan
            
            # 2. HAR-RV
            # Use realized volatility 'RV_1d' as target and features
            print("  Fitting HAR-RV...")
            har = HARRVModel()
            
            # Target: RV_1d (Daily Volatility)
            # Feature calculation handles NaNs internally by dropping
            target_rv = df['RV_1d'].dropna()
            
            if len(target_rv) > 100:
                har_res = har.fit(target_rv)
                har_r2 = har_res.rsquared
                print(f"  HAR-RV R2: {har_r2:.4f}")
            else:
                har_r2 = np.nan
            
            # Store Results
            results.append({
                'Symbol': symbol,
                'GARCH_Best': best_garch,
                'GARCH_AIC': garch_aic,
                'HAR_R2': har_r2
            })
            
        except Exception as e:
            print(f"Error on {symbol}: {e}")
            continue
            
    # Save Results
    results_df = pd.DataFrame(results)
    print("\n=== Baseline Results ===")
    print(results_df.to_markdown())
    results_df.to_csv('baseline_results.csv', index=False)

if __name__ == "__main__":
    run_baselines()

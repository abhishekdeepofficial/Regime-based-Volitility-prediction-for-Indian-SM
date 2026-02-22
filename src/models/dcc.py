import pandas as pd
import numpy as np
from arch import arch_model


def fit_dcc_proxy(returns_df: pd.DataFrame, window: int = 375) -> pd.Series:
    """
    Fit a Dynamic Conditional Correlation (DCC) proxy.
    
    True DCC is computationally expensive and complex to implement from scratch.
    A robust proxy is to:
    1. Fit univariate GARCH to each stock to get standardized residuals.
    2. Compute rolling correlation of standardized residuals.
    3. Average these correlations to get a "Market Correlation" index.
    
    Args:
        returns_df: DataFrame of returns for multiple stocks (aligned timestamps)
        window: Rolling window for correlation (375 bars = 1 week approx)
        
    Returns:
        pd.Series: Market-wide correlation index
    """
    print(f"Fits DCC proxy on {len(returns_df.columns)} stocks...")
    
    # 1. Standardize Residuals (GARCH(1,1))
    std_resid = pd.DataFrame(index=returns_df.index)
    
    for symbol in returns_df.columns:
        try:
            # Scale for stability
            y = returns_df[symbol].dropna() * 100
            
            if len(y) < 1000: # Skip short history
                continue
                
            am = arch_model(y, vol='GARCH', p=1, q=1, dist='normal')
            res = am.fit(disp='off', show_warning=False)
            
            # Get standardized residuals (z = r / sigma)
            # Reindex to match original df
            std_resid[symbol] = res.std_resid.reindex(returns_df.index)
            
        except Exception as e:
            print(f"DCC GARCH fit failed for {symbol}: {e}")
            
    # 2. Rolling Correlation
    # Compute average pairwise correlation
    # Faster approach: Compute rolling correlation matrix, then average off-diagonals
    
    # Actually, simplistic proxy: 
    # Average rolling correlation of each stock with the "Market" (mean of all stocks)
    # This is much faster and captures the "Market Regime" idea well.
    
    # Calculate Market Return (Equal Weighted)
    market_ret = returns_df.mean(axis=1)
    
    avg_corr = pd.Series(0.0, index=returns_df.index)
    count = 0
    
    for symbol in std_resid.columns:
        # Rolling corr with market
        r = std_resid[symbol].rolling(window=window).corr(market_ret)
        avg_corr = avg_corr.add(r, fill_value=0)
        count += 1
        
    market_correlation = avg_corr / count
    
    return market_correlation.fillna(method='ffill').fillna(0)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    from src.data.pipeline import load_all_stocks # Or load processed
    from config.settings import PROCESSED_DATA_DIR
    
    # Load processed returns
    print("Loading data for DCC...")
    processed_files = list(PROCESSED_DATA_DIR.glob("*_processed.parquet"))
    
    returns_map = {}
    for p in processed_files:
        if "pooled" in p.name: continue
        symbol = p.name.replace("_processed.parquet", "")
        df = pd.read_parquet(p)
        returns_map[symbol] = df.set_index('datetime')['deseasonalized_return']
        
    returns_df = pd.DataFrame(returns_map)
    
    # Fit
    dcc_idx = fit_dcc_proxy(returns_df)
    print("\nMarket Correlation Index:")
    print(dcc_idx.describe())
    
    # Save
    dcc_idx.to_csv(PROCESSED_DATA_DIR / "market_correlation.csv")

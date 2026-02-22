import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Any, Tuple

class GARCHModel:
    def __init__(self, p: int = 1, q: int = 1, dist: str = 'studentst'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.result = None
    
    def fit(self, returns: pd.Series, model_type: str = 'GARCH') -> Any:
        """
        Fit a GARCH family model.
        
        Args:
            returns: Series of returns (will be scaled by 100 for stability)
            model_type: 'GARCH', 'EGARCH', 'GJR-GARCH'
        """
        # Scale returns for better numerical stability in GARCH optimization
        scaled_returns = returns * 100
        
        if model_type == 'GARCH':
            self.model = arch_model(scaled_returns, vol='GARCH', p=self.p, q=self.q, dist=self.dist)
        elif model_type == 'EGARCH':
            self.model = arch_model(scaled_returns, vol='EGARCH', p=self.p, q=self.q, dist=self.dist)
        elif model_type == 'GJR-GARCH':
            # GJR-GARCH is GARCH with asymmetric shock (o=1)
            self.model = arch_model(scaled_returns, vol='GARCH', p=self.p, o=1, q=self.q, dist=self.dist)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.result = self.model.fit(disp='off', show_warning=False)
        return self.result

    def predict(self, horizon: int = 1) -> pd.Series:
        """
        Predict volatility.
        """
        if self.result is None:
            raise ValueError("Model must be fit first")
            
        # Forecast
        forecasts = self.result.forecast(horizon=horizon)
        
        # Extract variance and convert to volatility
        # Note: Forecast is variance of scaled returns (r*100)^2 -> vol is sqrt(var)/100
        var_forecast = forecasts.variance.values[-1, :]
        vol_forecast = np.sqrt(var_forecast) / 100
        
        return pd.Series(vol_forecast)

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get in-sample conditional volatility.
        """
        if self.result is None:
            raise ValueError("Model must be fit first")
            
        # Rescale back to original units
        return self.result.conditional_volatility / 100

def train_garch_benchmarks(returns: pd.Series) -> Dict[str, Dict]:
    """
    Train and compare multiple GARCH variants.
    """
    models = ['GARCH', 'EGARCH', 'GJR-GARCH']
    results = {}
    
    for m in models:
        try:
            garch = GARCHModel()
            res = garch.fit(returns, model_type=m)
            
            results[m] = {
                'aic': res.aic,
                'bic': res.bic,
                'params': res.params,
                'model_obj': garch
            }
        except Exception as e:
            print(f"Failed to fit {m}: {e}")
            
    return results

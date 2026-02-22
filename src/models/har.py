import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional

class HARRVModel:
    def __init__(self, lags: List[int] = [1, 5, 22]):
        """
        HAR-RV Model (Heterogeneous Autoregressive model of Realized Volatility).
        Default lags: 1 (daily), 5 (weekly), 22 (monthly).
        """
        self.lags = lags
        self.model = None
        self.result = None
        
    def prepare_features(self, rv_series: pd.Series) -> pd.DataFrame:
        """
        Create HAR features (lagged averages).
        """
        df = pd.DataFrame({'RV': rv_series})
        
        # RV_d (Yesterday)
        df['RV_d'] = df['RV'].shift(1)
        
        # RV_w (Last week avg)
        df['RV_w'] = df['RV'].rolling(window=5).mean().shift(1)
        
        # RV_m (Last month avg)
        df['RV_m'] = df['RV'].rolling(window=22).mean().shift(1)
        
        return df.dropna()

    def fit(self, rv_series: pd.Series, extended_features: Optional[pd.DataFrame] = None):
        """
        Fit HAR-RV model.
        Args:
            rv_series: Target Realized Volatility series (e.g., RV_1d)
            extended_features: Optional extra features (e.g., Jumps, Overnight Vol)
        """
        # Create base features
        X = self.prepare_features(rv_series)
        y = X['RV']
        X = X.drop('RV', axis=1)
        
        # Add extended features if any (aligned index)
        if extended_features is not None:
            X = X.join(extended_features, how='inner')
            y = y.loc[X.index]
            
        # Add constant
        X = sm.add_constant(X)
        
        # Fit OLS with Newey-West standard errors (HAC)
        self.model = sm.OLS(y, X)
        self.result = self.model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        return self.result

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted model.
        """
        if self.result is None:
            raise ValueError("Model must be fit first")
            
        features = sm.add_constant(features)
        return self.result.predict(features)

    def summary(self):
        if self.result:
            return self.result.summary()
        return "Model not fit"

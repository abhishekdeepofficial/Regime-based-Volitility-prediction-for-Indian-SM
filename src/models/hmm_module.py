import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib

class RegimeDetector:
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=100, 
            random_state=random_state
        )
        self.regime_map = {} # To map hidden states 0-3 to semantic names
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for HMM:
        1. Log Returns (Direction/Magnitude)
        2. Realized Volatility (Risk)
        """
        data = df[['log_return', 'RV_1d']].copy()
        
        # Handle outliers/infinities
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Scale features
        data = data * 100
        
        return data

    def fit(self, df: pd.DataFrame):
        X = self.prepare_features(df)
        self.model.fit(X)
        
        # Identify Regimes by Volatility
        means = self.model.means_
        state_stats = pd.DataFrame(means, columns=['Return', 'Volatility'])
        state_stats['State'] = range(self.n_components)
        
        sorted_stats = state_stats.sort_values('Volatility')
        sorted_states = sorted_stats['State'].values
        
        # Map sorted index to label
        self.regime_map = {
            sorted_states[0]: 'Low_Vol',
            sorted_states[1]: 'Normal_Vol',
            sorted_states[2]: 'High_Vol',
            sorted_states[3]: 'Extreme_Vol'
        }
        
        print("Regime Mapping (Vol-based):")
        print(state_stats.sort_values('Volatility'))
        
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.prepare_features(df)
        hidden_states = self.model.predict(X)
        
        result = df.loc[X.index].copy()
        result['regime_id'] = hidden_states
        result['regime'] = result['regime_id'].map(self.regime_map)
        
        return result

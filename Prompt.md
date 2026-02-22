🎯 Hybrid Regime-Aware Multiscale Volatility Prediction System
STOCK MARKET EDITION - Complete Technical Implementation Roadmap
🔄 KEY MODIFICATIONS FOR EQUITY MARKETS
Critical Differences from Crypto:
Market Hours: Trading sessions (9:15 AM - 3:30 PM IST) vs 24/7 crypto
Corporate Actions: Dividends, splits, bonuses require adjustment
Market Microstructure: Circuit breakers, tick sizes, auction mechanisms
Lower Volatility: Stock markets less volatile than crypto (different thresholds)
Multi-Asset Opportunities: 10 years of Reliance + other stocks = richer dataset
Regulatory Events: RBI policy, budget announcements, earnings
📊 RECOMMENDED MULTI-STOCK APPROACH
YES - You should absolutely add other stocks. Here's why:

Benefits of Multi-Asset Training:
Cross-Sectional Learning: Model learns volatility patterns across assets
Regime Generalization: Crisis periods (COVID, 2013 taper tantrum) affect all stocks
Data Augmentation: 10 stocks × 10 years = 100 stock-years of data
Sector Dynamics: Different sectors have different volatility profiles
Transfer Learning: Pre-train on multiple stocks, fine-tune on Reliance
Recommended Stock Selection (10-15 stocks):
Criteria: Liquid, representative sectors, long history

Stock	Sector	Volatility Profile	Rationale
Reliance	Energy/Telecom	Moderate	Primary asset
HDFC Bank	Banking	Low-Moderate	Defensive, liquid
Infosys/TCS	IT	Moderate	Export exposure
ITC	FMCG	Low	Stable, defensive
ICICI Bank	Banking	Moderate-High	Cyclical banking
Bharti Airtel	Telecom	Moderate	Sector peer to Reliance
L&T	Capital Goods	High	Infrastructure, cyclical
Maruti	Auto	High	Consumer cyclical
SBI	PSU Banking	Very High	High beta
Asian Paints	Consumer	Low-Moderate	Quality stock
Wipro	IT	Moderate	IT sector diversity
Sun Pharma	Pharma	Moderate	Healthcare exposure
Nifty 50 ETF	Index	Moderate	Market benchmark
Implementation Strategy:

Primary Model: Train on all 13 stocks (pooled)
Asset-Specific Fine-tuning: Fine-tune on Reliance for deployment
Hierarchical Model: Shared encoder + stock-specific heads
Phase 0 — Research Foundation & Literature Review (Weeks 1-2)
Core Literature Requirements
1. GARCH Family Models (Same as before)
GARCH(1,1): Bollerslev (1986)
EGARCH: Nelson (1991) - Critical for stocks (leverage effect stronger)
GJR-GARCH: Glosten et al. (1993)
NEW: FIGARCH - Fractionally Integrated GARCH for long memory
2. Realized Volatility Framework
Andersen & Bollerslev (1998)
NEW: Andersen et al. (2001) - "The Distribution of Realized Stock Return Volatility"
Documents empirical properties of equity RV
Shows RV is lognormally distributed
Barndorff-Nielsen & Shephard (2004): Bipower Variation
NEW: Hansen & Lunde (2006) - "Realized Variance and Market Microstructure Noise"
Optimal sampling frequency for stocks (5-min is good)
Noise-robust estimators
3. Intraday Patterns (CRITICAL FOR STOCKS)
Andersen & Bollerslev (1997): "Intraday Periodicity and Volatility Persistence"
U-shaped intraday volatility pattern (high at open/close)
Must deseasonalize for accurate forecasting
NEW: Taylor & Xu (1997): "The Incremental Volatility Information in One Million FX Quotations"
Seasonal adjustment methods
4. Corporate Actions & Market Microstructure
Dubofsky & Groth (1984): Stock splits effect on volatility
Bajaj & Vijh (1990): Dividend announcements
NEW: Sensoy & Tabak (2016): "Dynamic efficiency of stock markets and exchange rates"
Regime changes around major events
5. Multi-Asset Volatility Modeling
Engle (2002): Dynamic Conditional Correlation (DCC-GARCH)
Bauwens et al. (2006): "Multivariate GARCH models: a survey"
NEW: Brownlees & Gallo (2010): "Comparison of Volatility Measures: a Risk Management Perspective"
6. Indian Market Specific
NEW: Karmakar (2005): "Modeling Conditional Volatility of the Indian Stock Markets"
GARCH specifications for Indian equities
NEW: Bansal & Dahlquist (2002): "Expropriation risk and return in global equity markets"
Emerging market volatility characteristics
Deliverables
Extended literature review (20-25 pages) including equity-specific citations
Hypothesis refinement:
"Multi-asset trained hybrid models with intraday seasonality adjustment outperform single-asset models by >20% QLIKE during crisis regimes"
Baseline specifications:
EGARCH(1,1) with GED distribution (better for stock returns)
HAR-RV with intraday adjustments
DCC-GARCH for multi-asset correlation
Phase 1 — Data Engineering Pipeline (Weeks 3-6)
1.1 Data Acquisition & Organization
Data Structure
python
import pandas as pd
import numpy as np
from pathlib import Path

# Organize data by stock
data_dir = Path("data/raw/")
stocks = [
    'RELIANCE', 'HDFCBANK', 'INFY', 'ITC', 'ICICIBANK',
    'BHARTIARTL', 'LT', 'MARUTI', 'SBIN', 'ASIANPAINT',
    'WIPRO', 'SUNPHARMA', 'NIFTY50'
]

# Load function
def load_stock_data(symbol, start_date='2014-01-01', end_date='2024-12-31'):
    """
    Load 5-min OHLCV data for a stock
    Expected format: CSV with columns [datetime, open, high, low, close, volume]
    """
    df = pd.read_csv(data_dir / f"{symbol}_5min.csv", parse_dates=['datetime'])
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# Load all stocks
data = {symbol: load_stock_data(symbol) for symbol in stocks}
Data Requirements
Frequency: 5-minute bars
Period: 2014-2024 (10 years)
Bars per day: 75 bars (6.5 hours × 12 five-min intervals/hour)
Total bars per stock: ~187,500 (75 × 250 trading days × 10 years)
Format: datetime, open, high, low, close, volume, open_interest (if futures)
1.2 Corporate Actions Adjustment ⭐⭐⭐ CRITICAL
python
def adjust_for_corporate_actions(df, symbol):
    """
    Adjust prices for splits, bonuses, dividends
    Use adjustment factor approach (like NSE bhavcopy)
    """
    # Load corporate actions from NSE or manual file
    actions = load_corporate_actions(symbol)
    
    # Calculate cumulative adjustment factor
    df['adj_factor'] = 1.0
    
    for _, action in actions.iterrows():
        action_date = action['date']
        
        if action['type'] == 'split':
            # Example: 1:2 split means prices should be halved post-split
            ratio = action['old_fv'] / action['new_fv']
            df.loc[df['datetime'] < action_date, 'adj_factor'] *= ratio
            
        elif action['type'] == 'bonus':
            # Example: 1:1 bonus means prices halved
            ratio = (action['old_shares'] + action['bonus_shares']) / action['old_shares']
            df.loc[df['datetime'] < action_date, 'adj_factor'] *= ratio
            
        elif action['type'] == 'dividend':
            # Adjust for ex-dividend price drop
            div_yield = action['dividend'] / df.loc[df['datetime'] == action_date, 'close'].iloc[0]
            df.loc[df['datetime'] < action_date, 'adj_factor'] *= (1 + div_yield)
    
    # Apply adjustments
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_adj'] = df[col] / df['adj_factor']
    
    return df

# Apply to all stocks
for symbol in stocks:
    data[symbol] = adjust_for_corporate_actions(data[symbol], symbol)
1.3 Market Hours Filtering
python
def filter_trading_hours(df):
    """
    Keep only regular trading hours: 09:15 - 15:30 IST
    Remove pre-open, post-close
    """
    df['time'] = df['datetime'].dt.time
    
    # Indian market hours
    market_open = pd.Timestamp('09:15').time()
    market_close = pd.Timestamp('15:30').time()
    
    df = df[(df['time'] >= market_open) & (df['time'] <= market_close)]
    
    return df.drop('time', axis=1)

# Apply to all
for symbol in stocks:
    data[symbol] = filter_trading_hours(data[symbol])
1.4 Data Cleaning Protocol
Step 1: Handle Market Holidays & Missing Data
python
def handle_missing_data(df):
    """
    Stock markets have holidays - handle gaps differently than crypto
    """
    # Create complete trading calendar (exclude weekends, holidays)
    from pandas.tseries.holiday import IndianHolidays
    
    # Generate all 5-min timestamps for trading days
    trading_days = pd.bdate_range(
        start=df['datetime'].min().date(),
        end=df['datetime'].max().date(),
        freq='B',  # Business days
        # Exclude NSE holidays - you'll need to provide this list
    )
    
    # Generate 5-min grid for each trading day
    trading_times = pd.date_range('09:15', '15:30', freq='5T').time
    expected_index = pd.DatetimeIndex([
        pd.Timestamp.combine(day, time)
        for day in trading_days
        for time in trading_times
    ])
    
    # Reindex to find missing bars
    df = df.set_index('datetime').reindex(expected_index).reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Forward-fill small gaps (<15 min = 3 bars)
    df = df.fillna(method='ffill', limit=3)
    
    # Flag days with >10% missing data
    df['date'] = df['datetime'].dt.date
    missing_pct = df.groupby('date')['close'].apply(lambda x: x.isna().mean())
    bad_days = missing_pct[missing_pct > 0.1].index
    
    # Remove bad days entirely
    df = df[~df['date'].isin(bad_days)]
    
    return df.drop('date', axis=1)

for symbol in stocks:
    data[symbol] = handle_missing_data(data[symbol])
Step 2: Remove Bad Ticks & Outliers
python
def clean_bad_ticks(df):
    """
    Remove stuck prices, zero volumes, extreme outliers
    """
    # Remove zero volume bars (likely bad data)
    df = df[df['volume'] > 0]
    
    # Remove stuck prices (same OHLC for >5 consecutive bars)
    df['price_change'] = df['close'].diff().abs()
    stuck = df['price_change'].rolling(5).sum() == 0
    df = df[~stuck]
    
    # Remove violations of OHLC logic
    df = df[
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ]
    
    # Outlier detection on returns (circuit filter should catch most)
    returns = np.log(df['close_adj'] / df['close_adj'].shift(1))
    Q1, Q3 = returns.quantile([0.01, 0.99])  # More aggressive for 5-min
    IQR = Q3 - Q1
    
    lower = Q1 - 5 * IQR
    upper = Q3 + 5 * IQR
    
    # Winsorize instead of removing (preserves time grid)
    returns = returns.clip(lower, upper)
    df['log_return'] = returns
    
    return df.drop('price_change', axis=1)

for symbol in stocks:
    data[symbol] = clean_bad_ticks(data[symbol])
1.5 Intraday Seasonality Adjustment ⭐⭐⭐ UNIQUE TO STOCKS
python
def deseasonalize_intraday(df):
    """
    Remove U-shaped intraday volatility pattern
    Following Andersen & Bollerslev (1997)
    """
    # Extract intraday time
    df['time_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    
    # Calculate seasonal pattern (average volatility by time)
    df['abs_return'] = df['log_return'].abs()
    seasonal_pattern = df.groupby('time_of_day')['abs_return'].transform('mean')
    
    # Normalize returns
    df['deseasonalized_return'] = df['log_return'] / (seasonal_pattern / seasonal_pattern.mean())
    
    # Store seasonal factor for later re-seasonalization of forecasts
    df['seasonal_factor'] = seasonal_pattern / seasonal_pattern.mean()
    
    return df

for symbol in stocks:
    data[symbol] = deseasonalize_intraday(data[symbol])
1.6 Feature Engineering — Multiscale Volatility
Core Realized Measures
python
def compute_realized_volatility(returns, window=75, annualize=True):
    """
    Realized Volatility for stock market
    window=75 = 1 trading day
    Annualize using sqrt(252) for trading days
    """
    rv = returns.rolling(window).apply(lambda x: np.sqrt((x**2).sum()))
    if annualize:
        rv *= np.sqrt(252)
    return rv

# For each stock
for symbol in stocks:
    df = data[symbol]
    
    # Use deseasonalized returns for RV computation
    returns = df['deseasonalized_return']
    
    # Multiscale RV
    df['RV_30m'] = compute_realized_volatility(returns, window=6)   # 30 min
    df['RV_1h'] = compute_realized_volatility(returns, window=12)  # 1 hour
    df['RV_half_day'] = compute_realized_volatility(returns, window=38)  # Half day
    df['RV_1d'] = compute_realized_volatility(returns, window=75)  # Full day
    df['RV_1w'] = compute_realized_volatility(returns, window=75*5)  # 1 week
    
    data[symbol] = df
Jump Detection (Enhanced for Circuit Breakers)
python
def detect_jumps_with_circuits(df):
    """
    Jump detection accounting for circuit breakers (10%/20% limits in India)
    """
    returns = df['log_return']
    
    # Bipower Variation
    abs_returns = returns.abs()
    bv = (np.pi/2) * (abs_returns * abs_returns.shift(1)).rolling(75).sum()
    df['BV_1d'] = np.sqrt(bv * 252)
    
    # Jump component
    df['Jump_1d'] = np.maximum(df['RV_1d'] - df['BV_1d'], 0)
    df['Jump_ratio'] = df['Jump_1d'] / df['RV_1d']
    
    # Circuit breaker flag (absolute return > 9%)
    df['circuit_breaker'] = (returns.abs() > 0.09).astype(int)
    
    # Jump significance (more conservative for stocks)
    df['Jump_sig'] = ((df['Jump_ratio'] > 0.15) | (df['circuit_breaker'] == 1)).astype(int)
    
    return df

for symbol in stocks:
    data[symbol] = detect_jumps_with_circuits(data[symbol])
Range-Based Estimators
python
def compute_range_estimators(df):
    """
    Parkinson, Garman-Klass, Rogers-Satchell estimators
    """
    # Parkinson (assumes zero drift)
    hl_ratio = np.log(df['high_adj'] / df['low_adj'])
    df['Parkinson_1d'] = np.sqrt(
        (hl_ratio**2).rolling(75).mean() / (4 * np.log(2)) * 252
    )
    
    # Garman-Klass (more efficient)
    hl = np.log(df['high_adj'] / df['low_adj'])**2
    co = np.log(df['close_adj'] / df['open_adj'])**2
    df['GK_1d'] = np.sqrt(
        (0.5 * hl - (2*np.log(2)-1) * co).rolling(75).mean() * 252
    )
    
    # Rogers-Satchell (allows drift)
    hl = np.log(df['high_adj'] / df['close_adj']) * np.log(df['high_adj'] / df['open_adj'])
    ll = np.log(df['low_adj'] / df['close_adj']) * np.log(df['low_adj'] / df['open_adj'])
    df['RS_1d'] = np.sqrt((hl + ll).rolling(75).mean() * 252)
    
    return df

for symbol in stocks:
    data[symbol] = compute_range_estimators(data[symbol])
Volume & Liquidity Features
python
def compute_volume_features(df):
    """
    Volume-based features (more important for stocks than crypto)
    """
    # Volume MA and volatility
    df['volume_ma'] = df['volume'].rolling(75).mean()
    df['volume_std'] = df['volume'].rolling(75).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (df['volume_std'] + 1e-8)
    
    # Amihud illiquidity measure
    df['illiquidity'] = df['log_return'].abs() / (df['volume'] * df['close_adj'] + 1e-8)
    df['illiquidity_ma'] = df['illiquidity'].rolling(75).mean()
    
    # Volume-weighted volatility
    df['vwap'] = (df['close_adj'] * df['volume']).rolling(75).sum() / df['volume'].rolling(75).sum()
    df['vwap_distance'] = (df['close_adj'] - df['vwap']) / df['vwap']
    
    # Roll measure (implicit spread estimator)
    cov_ret = df['log_return'].rolling(2).apply(lambda x: np.cov(x[:-1], x[1:])[0,1])
    df['roll_spread'] = 2 * np.sqrt(-cov_ret.clip(upper=0).abs())
    
    return df

for symbol in stocks:
    data[symbol] = compute_volume_features(data[symbol])
Temporal Encoding (Market-Specific)
python
def add_temporal_features(df):
    """
    Time-based features specific to stock markets
    """
    # Intraday time
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_of_day'] = df['hour'] * 60 + df['minute']
    
    # Day of week (Monday effect, Friday effect)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Month (January effect, year-end effects)
    df['month'] = df['datetime'].dt.month
    
    # Quarter (earnings season)
    df['quarter'] = df['datetime'].dt.quarter
    
    # Days to month/quarter end (window dressing)
    df['days_to_month_end'] = df['datetime'].dt.days_in_month - df['datetime'].dt.day
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)  # 5 trading days
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Opening/closing auction flags
    df['is_opening'] = (df['time_of_day'] <= 555).astype(int)  # First 30 min
    df['is_closing'] = (df['time_of_day'] >= 900).astype(int)  # Last 30 min
    
    return df

for symbol in stocks:
    data[symbol] = add_temporal_features(data[symbol])
Cross-Asset Features ⭐ NEW - Multi-Stock Model
python
def compute_cross_asset_features(data_dict):
    """
    Features leveraging multiple stocks
    """
    # Compute market-wide volatility (Nifty 50)
    market_vol = data_dict['NIFTY50']['RV_1d']
    
    # Add market volatility to each stock
    for symbol in data_dict.keys():
        if symbol == 'NIFTY50':
            continue
        
        df = data_dict[symbol]
        
        # Align timestamps
        df = df.merge(
            market_vol.to_frame('market_vol'),
            left_on='datetime',
            right_index=True,
            how='left'
        )
        
        # Relative volatility (stock vs market)
        df['vol_ratio'] = df['RV_1d'] / (df['market_vol'] + 1e-8)
        
        # Beta (rolling correlation with market)
        stock_ret = df['log_return']
        market_ret = data_dict['NIFTY50'].set_index('datetime')['log_return']
        
        # Align
        aligned = pd.concat([stock_ret, market_ret], axis=1, keys=['stock', 'market']).dropna()
        
        # Rolling beta
        window = 75 * 20  # 20 days
        df['beta'] = aligned['stock'].rolling(window).cov(aligned['market']) / \
                     aligned['market'].rolling(window).var()
        
        data_dict[symbol] = df
    
    return data_dict

data = compute_cross_asset_features(data)
1.7 Target Variable Construction
python
def create_targets(df, horizons=[12, 48, 75]):
    """
    Create forward-looking volatility targets
    horizons: [1h, 4h, 1d] in 5-min bars
    """
    for h in horizons:
        # Forward RV
        df[f'target_RV_{h}bar'] = df['RV_1d'].shift(-h)
        
        # Re-seasonalize for actual prediction
        # (model predicts deseasonalized, convert back at inference)
        avg_seasonal = df['seasonal_factor'].shift(-h).rolling(h).mean()
        df[f'target_RV_{h}bar_seasonal'] = df[f'target_RV_{h}bar'] * avg_seasonal
    
    # Remove rows without targets
    df = df.dropna(subset=[f'target_RV_{horizons[0]}bar'])
    
    return df

for symbol in stocks:
    data[symbol] = create_targets(data[symbol])
1.8 Create Unified Multi-Asset Dataset
python
def create_pooled_dataset(data_dict):
    """
    Combine all stocks into one dataset for multi-asset training
    """
    all_data = []
    
    for symbol, df in data_dict.items():
        df = df.copy()
        df['symbol'] = symbol
        all_data.append(df)
    
    pooled = pd.concat(all_data, ignore_index=True)
    
    # Add stock-specific encoding (for embedding layer)
    symbol_map = {s: i for i, s in enumerate(data_dict.keys())}
    pooled['symbol_id'] = pooled['symbol'].map(symbol_map)
    
    # Save
    pooled.to_parquet('data/processed/pooled_data.parquet')
    
    return pooled

pooled_data = create_pooled_dataset(data)
print(f"Total samples: {len(pooled_data):,}")
print(f"Stocks: {pooled_data['symbol'].nunique()}")
print(f"Date range: {pooled_data['datetime'].min()} to {pooled_data['datetime'].max()}")
Deliverables
Cleaned datasets: Individual parquet files per stock + pooled dataset
Corporate actions log: All adjustments documented
Data quality report:
Missing data % by stock
Outlier removal count
Circuit breaker events
Summary statistics by stock
Feature correlation matrix: Check multicollinearity (separate for each stock + pooled)
Intraday seasonality plots: U-shaped pattern visualization
Phase 2 — Baseline Statistical Models (Weeks 7-8)
2.1 GARCH Models (Stock-Specific)
python
from arch import arch_model

def fit_garch_models(df, distribution='studentst'):
    """
    Fit GARCH family to stock returns
    StudentT distribution recommended for stock fat tails
    """
    # Use deseasonalized returns (percentage for stability)
    returns = df['deseasonalized_return'].dropna() * 100
    
    models = {}
    
    # GARCH(1,1)
    models['GARCH'] = arch_model(returns, vol='GARCH', p=1, q=1, dist=distribution)
    
    # EGARCH(1,1) - captures leverage effect (very important for stocks)
    models['EGARCH'] = arch_model(returns, vol='EGARCH', p=1, q=1, dist=distribution)
    
    # GJR-GARCH(1,1,1)
    models['GJR-GARCH'] = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist=distribution)
    
    # Fit all models
    results = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        res = model.fit(disp='off', show_warning=False)
        results[name] = {
            'fit': res,
            'AIC': res.aic,
            'BIC': res.bic,
            'volatility': res.conditional_volatility / 100
        }
    
    # Select best by AIC
    best_model = min(results, key=lambda x: results[x]['AIC'])
    print(f"Best model: {best_model}")
    
    return results, best_model

# Fit for each stock
garch_results = {}
for symbol in stocks:
    print(f"\n=== {symbol} ===")
    results, best = fit_garch_models(data[symbol])
    garch_results[symbol] = results[best]
    
    # Add to dataframe
    data[symbol]['garch_vol'] = results[best]['volatility']
    data[symbol]['garch_resid'] = results[best]['fit'].std_resid
2.2 HAR-RV Model with Intraday Components
python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def fit_har_rv_extended(df):
    """
    HAR-RV with daily, weekly, monthly + intraday components
    Following Andersen et al. (2007)
    """
    # Prepare features
    df['RV_lag1'] = df['RV_1d'].shift(75)  # Yesterday
    df['RV_weekly'] = df['RV_1d'].rolling(75*5).mean().shift(75)  # Last week avg
    df['RV_monthly'] = df['RV_1d'].rolling(75*22).mean().shift(75)  # Last month avg
    
    # Overnight volatility (close-to-open)
    df['overnight_vol'] = np.log(df['open_adj'] / df['close_adj'].shift(75)).abs()
    
    # Intraday components
    df['RV_morning'] = df['RV_30m'].shift(38)  # Morning session vol
    df['RV_afternoon'] = df['RV_30m'].shift(6)  # Afternoon vol
    
    # Jump component
    df['Jump_lag1'] = df['Jump_1d'].shift(75)
    
    # Setup regression
    features = [
        'RV_lag1', 'RV_weekly', 'RV_monthly',
        'overnight_vol', 'RV_morning', 'RV_afternoon',
        'Jump_lag1'
    ]
    
    X = df[features].dropna()
    X = add_constant(X)
    y = df.loc[X.index, 'target_RV_75bar']
    
    # Fit with HAC standard errors (Newey-West)
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
    
    print(model.summary())
    
    df.loc[X.index, 'har_forecast'] = model.predict(X)
    
    return model, df

# Fit for each stock
har_models = {}
for symbol in stocks:
    print(f"\n=== HAR-RV for {symbol} ===")
    model, data[symbol] = fit_har_rv_extended(data[symbol])
    har_models[symbol] = model
2.3 DCC-GARCH for Multi-Asset ⭐ NEW
python
from arch.univariate import ConstantMean, GARCH
from arch.univariate import Normal
import warnings
warnings.filterwarnings('ignore')

def fit_dcc_garch(returns_matrix):
    """
    Dynamic Conditional Correlation GARCH
    Model time-varying correlations between stocks
    
    This provides market-level volatility features
    """
    # Step 1: Fit univariate GARCH to each stock
    n_stocks = returns_matrix.shape[1]
    residuals = np.zeros_like(returns_matrix)
    std_residuals = np.zeros_like(returns_matrix)
    
    for i in range(n_stocks):
        ret = returns_matrix.iloc[:, i].dropna()
        am = ConstantMean(ret * 100)
        am.volatility = GARCH(p=1, q=1)
        am.distribution = Normal()
        res = am.fit(disp='off')
        
        # Extract standardized residuals
        std_residuals[:, i] = res.std_resid.reindex(returns_matrix.index, fill_value=0)
    
    # Step 2: DCC model on standardized residuals
    # (Simplified - full DCC requires optimization of correlation dynamics)
    # For now, use rolling correlation as proxy
    
    rolling_corr = pd.DataFrame(std_residuals).rolling(75*5).corr()
    
    # Average correlation with market
    avg_correlation = rolling_corr.groupby(level=0).mean().mean(axis=1)
    
    return avg_correlation

# Prepare returns matrix
returns_matrix = pd.DataFrame({
    symbol: data[symbol].set_index('datetime')['deseasonalized_return']
    for symbol in stocks[:5]  # Use top 5 for speed
})

market_correlation = fit_dcc_garch(returns_matrix)

# Add to each stock
for symbol in stocks:
    data[symbol] = data[symbol].merge(
        market_correlation.to_frame('market_correlation'),
        left_on='datetime',
        right_index=True,
        how='left'
    )
Deliverables
GARCH results by stock: AIC/BIC comparison table
HAR-RV coefficients: Significance of daily/weekly/monthly components
DCC-GARCH output: Market correlation dynamics
Model comparison: Out-of-sample RMSE/QLIKE for each baseline
Phase 3 — Regime Detection System (Weeks 9-10)
3.1 Multi-State HMM with Market Events
python
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

def fit_regime_hmm(df, n_regimes=4):
    """
    4-state HMM for stocks:
    0: Low volatility (calm markets)
    1: Normal volatility
    2: High volatility (pre-crisis, elevated uncertainty)
    3: Crisis (extreme events)
    """
    # Features for regime identification
    regime_features = df[[
        'RV_1d', 'Jump_1d', 'volume_zscore', 
        'illiquidity', 'overnight_vol'
    ]].dropna()
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(regime_features)
    
    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=1000,
        random_state=42,
        tol=1e-4
    )
    
    model.fit(X)
    
    # Predict regimes
    regimes = model.predict(X)
    df.loc[regime_features.index, 'regime_hmm'] = regimes
    
    # Order regimes by mean volatility
    regime_means = df.groupby('regime_hmm')['RV_1d'].mean().sort_values()
    regime_map = {old: new for new, old in enumerate(regime_means.index)}
    df['regime'] = df['regime_hmm'].map(regime_map)
    
    # Label regimes
    df['regime_label'] = df['regime'].map({
        0: 'Low Vol',
        1: 'Normal',
        2: 'High Vol',
        3: 'Crisis'
    })
    
    return model, df

# Fit for each stock
regime_models = {}
for symbol in stocks:
    print(f"\n=== Regime Detection: {symbol} ===")
    model, data[symbol] = fit_regime_hmm(data[symbol])
    regime_models[symbol] = model
    
    # Print regime statistics
    print(data[symbol].groupby('regime_label')['RV_1d'].describe())
3.2 Event-Based Regime Validation
python
def validate_regimes_with_events(df, symbol):
    """
    Check if crisis regimes align with known events
    """
    # Known crisis periods in Indian markets
    crisis_events = {
        '2020 COVID Crash': ('2020-03-01', '2020-04-30'),
        '2018 IL&FS Crisis': ('2018-09-01', '2018-10-31'),
        '2016 Demonetization': ('2016-11-08', '2016-12-31'),
        '2013 Taper Tantrum': ('2013-05-22', '2013-08-31'),
    }
    
    for event_name, (start, end) in crisis_events.items():
        mask = (df['datetime'] >= start) & (df['datetime'] <= end)
        if mask.sum() > 0:
            crisis_pct = (df.loc[mask, 'regime'] == 3).mean()
            print(f"{symbol} - {event_name}: {crisis_pct:.1%} in Crisis regime")

for symbol in stocks:
    validate_regimes_with_events(data[symbol], symbol)
3.3 Market-Wide Regime Synchronization
python
def compute_market_regime(data_dict):
    """
    Aggregate individual stock regimes into market regime
    """
    # For each timestamp, count how many stocks are in each regime
    regime_counts = pd.DataFrame()
    
    for symbol in data_dict.keys():
        df = data_dict[symbol][['datetime', 'regime']].copy()
        df = df.rename(columns={'regime': symbol})
        
        if regime_counts.empty:
            regime_counts = df.set_index('datetime')
        else:
            regime_counts = regime_counts.join(df.set_index('datetime'), how='outer')
    
    # Market regime = mode across stocks
    regime_counts['market_regime'] = regime_counts.mode(axis=1)[0]
    
    # Add back to individual stocks
    for symbol in data_dict.keys():
        data_dict[symbol] = data_dict[symbol].merge(
            regime_counts[['market_regime']],
            left_on='datetime',
            right_index=True,
            how='left'
        )
    
    return data_dict

data = compute_market_regime(data)
Deliverables
Regime labels for all stocks
Transition matrices by stock
Event validation report: Crisis regime alignment with known events
Market regime visualization: Timeline showing synchronized market states
Phase 4 — Deep Learning: Temporal Fusion Transformer (Weeks 11-14)
4.1 Multi-Asset TFT Architecture
python
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

def prepare_multi_asset_dataset(pooled_df, max_encoder_length=75):
    """
    Prepare dataset for multi-asset TFT training
    """
    # Create time index (continuous across all stocks)
    pooled_df['time_idx'] = (
        pooled_df['datetime'] - pooled_df['datetime'].min()
    ).dt.total_seconds() / 300  # 5-min units
    pooled_df['time_idx'] = pooled_df['time_idx'].astype(int)
    
    # Static features (stock-level)
    static_categoricals = ['symbol']
    static_reals = []
    
    # Known future features
    time_varying_known_categoricals = ['is_opening', 'is_closing', 'quarter']
    time_varying_known_reals = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
        'month_sin', 'month_cos', 'days_to_month_end'
    ]
    
    # Unknown future features (to be predicted)
    time_varying_unknown_categoricals = ['regime', 'circuit_breaker']
    time_varying_unknown_reals = [
        'deseasonalized_return',
        'RV_30m', 'RV_1h', 'RV_half_day', 'RV_1d',
        'BV_1d', 'Jump_1d',
        'Parkinson_1d', 'GK_1d', 'RS_1d',
        'garch_vol', 'garch_resid',
        'volume_zscore', 'illiquidity_ma',
        'vwap_distance', 'roll_spread',
        'market_vol', 'vol_ratio', 'beta',
        'market_correlation', 'overnight_vol'
    ]
    
    # Target
    target = 'target_RV_75bar'  # 1-day ahead
    
    # Split data (80% train, 20% validation by time)
    training_cutoff = pooled_df['time_idx'].quantile(0.8)
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        pooled_df[pooled_df['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target=target,
        group_ids=['symbol'],  # Separate time series per stock
        
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=75,  # Up to 1 day
        
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        
        # Normalization (per stock, to handle different scales)
        target_normalizer=GroupNormalizer(
            groups=['symbol'],
            transformation='log1p'  # Log(1+x) for positive targets
        ),
        
        categorical_encoders={'symbol': NaNLabelEncoder(add_nan=True)},
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        pooled_df[pooled_df['time_idx'] > training_cutoff],
        predict=True,
        stop_randomization=True
    )
    
    return training, validation

# Create datasets
training, validation = prepare_multi_asset_dataset(pooled_data)

# DataLoaders
batch_size = 64  # Smaller for multi-asset (more memory intensive)
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=8
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 2, num_workers=8
)
4.2 TFT Model with Stock Embeddings
python
# Initialize TFT
tft = TemporalFusionTransformer.from_dataset(
    training,
    
    # Architecture
    hidden_size=160,  # Larger for multi-asset
    lstm_layers=2,
    attention_head_size=4,
    dropout=0.15,
    hidden_continuous_size=32,
    
    # Embeddings
    embedding_sizes={
        'symbol': (len(stocks), 16),  # Stock embedding dimension
        'regime': (4, 8),  # Regime embedding
    },
    
    # Loss (QLIKE)
    loss=QLIKELoss(),  # From Phase 4 of crypto version
    
    # Optimizer
    learning_rate=5e-4,
    reduce_on_plateau_patience=4,
    
    # Logging
    log_interval=50,
    log_val_interval=1
)

print(f"Model parameters: {tft.size()/1e6:.2f}M")
4.3 Training with Multi-Asset Data
python
# Callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='models/tft_multiasset',
    filename='tft-{epoch:02d}-{val_loss:.4f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    gradient_clip_val=0.5,
    
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    
    logger=pl.loggers.TensorBoardLogger(
        'lightning_logs', 
        name='tft_multiasset_stocks'
    ),
    
    # Mixed precision for speed
    precision=16
)

# Train
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# Load best model
best_tft = TemporalFusionTransformer.load_from_checkpoint(
    checkpoint_callback.best_model_path
)
4.4 Stock-Specific Fine-Tuning ⭐ KEY STEP
python
def fine_tune_for_stock(base_model, symbol, data_dict, epochs=20):
    """
    Fine-tune multi-asset model on single stock for deployment
    """
    # Prepare stock-specific dataset
    stock_df = data_dict[symbol].copy()
    stock_df['time_idx'] = (
        stock_df['datetime'] - stock_df['datetime'].min()
    ).dt.total_seconds() / 300
    stock_df['time_idx'] = stock_df['time_idx'].astype(int)
    
    training_cutoff = stock_df['time_idx'].quantile(0.8)
    
    training_stock = TimeSeriesDataSet(
        stock_df[stock_df['time_idx'] <= training_cutoff],
        # ... (same config as before but with single stock)
        group_ids=['symbol'],
        # other params same
    )
    
    validation_stock = TimeSeriesDataSet.from_dataset(
        training_stock,
        stock_df[stock_df['time_idx'] > training_cutoff],
        predict=True
    )
    
    # Create new model from base with frozen encoder
    fine_tuned_model = TemporalFusionTransformer.from_dataset(
        training_stock,
        # Copy architecture from base
        **base_model.hparams
    )
    
    # Load pretrained weights
    fine_tuned_model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Freeze encoder layers (only train decoder)
    for name, param in fine_tuned_model.named_parameters():
        if 'static_encoder' in name or 'encoder' in name:
            param.requires_grad = False
    
    # Fine-tune
    trainer_ft = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'models/tft_{symbol}',
                filename='tft-finetuned-{epoch:02d}',
                save_top_k=1
            )
        ]
    )
    
    trainer_ft.fit(
        fine_tuned_model,
        training_stock.to_dataloader(train=True, batch_size=32),
        validation_stock.to_dataloader(train=False, batch_size=64)
    )
    
    return fine_tuned_model

# Fine-tune for Reliance (primary deployment target)
reliance_model = fine_tune_for_stock(best_tft, 'RELIANCE', data)
Deliverables
Multi-asset TFT model trained on all stocks
Fine-tuned Reliance model for production
Training curves: Loss by stock (to identify problematic assets)
Feature importance: Attention weights by stock
Cross-stock validation: Test Reliance model on HDFC, check transfer
Phase 5 — Hybrid GARCH-TFT Model (Week 15)
Same as crypto version - no major changes needed

Phase 6 — Evaluation (Week 16)
Additional Stock-Specific Metrics
python
def evaluate_stock_volatility_forecast(actuals, predictions, regime_labels=None):
    """
    Comprehensive evaluation for stock volatility forecasts
    """
    # Base metrics (same as crypto)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # QLIKE
    qlike = np.mean(np.log(predictions) + (actuals**2) / predictions)
    
    # MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Theil's U
    naive = actuals.shift(1).fillna(actuals.mean())
    u_stat = np.sqrt(mean_squared_error(actuals, predictions)) / \
             np.sqrt(mean_squared_error(actuals, naive))
    
    # NEW: Directional accuracy (for volatility increases/decreases)
    actual_direction = (actuals.diff() > 0).astype(int)
    pred_direction = (predictions.diff() > 0).astype(int)
    direction_accuracy = (actual_direction == pred_direction).mean()
    
    # NEW: Volatility quartile accuracy (classify into low/med/high/very high)
    actual_quartiles = pd.qcut(actuals, q=4, labels=False)
    pred_quartiles = pd.qcut(predictions, q=4, labels=False)
    quartile_accuracy = (actual_quartiles == pred_quartiles).mean()
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'QLIKE': qlike,
        'MAPE': mape,
        'Theil_U': u_stat,
        'Direction_Accuracy': direction_accuracy,
        'Quartile_Accuracy': quartile_accuracy
    }
    
    # Regime-specific metrics
    if regime_labels is not None:
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            if mask.sum() > 10:  # Minimum sample size
                metrics[f'QLIKE_Regime_{regime}'] = np.mean(
                    np.log(predictions[mask]) + (actuals[mask]**2) / predictions[mask]
                )
    
    return metrics

# Evaluate
for symbol in stocks:
    print(f"\n=== {symbol} ===")
    
    models_to_eval = {
        'GARCH': data[symbol]['garch_vol'],
        'HAR-RV': data[symbol]['har_forecast'],
        'TFT': tft_predictions[symbol],
        'Hybrid': hybrid_predictions[symbol]
    }
    
    results = {
        name: evaluate_stock_volatility_forecast(
            data[symbol]['target_RV_75bar'],
            preds,
            regime_labels=data[symbol]['regime']
        )
        for name, preds in models_to_eval.items()
    }
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_markdown())
Phase 7-10 — Same as Crypto Version
Hyperparameter optimization, real-time system, research paper, deployment follow same structure

Real-Time Data Source for Indian Stocks
python
# Replace Binance WebSocket with NSE/MCX streaming

# Option 1: Use broker APIs (Zerodha Kite, Interactive Brokers)
from kiteconnect import KiteTicker

kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    # Process tick data
    for tick in ticks:
        # tick contains: instrument_token, last_price, volume, etc.
        process_tick(tick)

kws.on_ticks = on_ticks
kws.connect(threaded=True)

# Option 2: Use NSE official data (delayed, not real-time for free users)
# Option 3: Commercial data providers (Bloomberg, Refinitiv)
📊 UPDATED TIMELINE
Phase	Duration	Key Deliverable
0. Research	2 weeks	Literature + stock-specific papers
1. Data Engineering	4 weeks	Multi-stock dataset + corp actions
2. GARCH Baseline	2 weeks	Per-stock models + DCC
3. Regime Detection	2 weeks	HMM + event validation
4. Deep Learning	4 weeks	Multi-asset TFT + fine-tuning
5. Hybrid Model	1 week	Combined architecture
6. Evaluation	1 week	Stock-specific metrics
7. Optimization	1 week	Hyperparameter tuning
8. Real-Time System	2 weeks	Broker API integration
9. Research Paper	2 weeks	Draft with multi-asset novelty
10. Deployment	2 weeks	Production system
Total	23 weeks (~5.5 months)	
🏆 FINAL SYSTEM FOR STOCKS
Your system will handle:

✅ Multi-asset learning from 13 stocks (10 years each = 130 stock-years)
✅ Corporate action adjustments (splits, dividends, bonuses)
✅ Intraday seasonality (U-shaped volatility pattern)
✅ Market hours (9:15 AM - 3:30 PM IST, no 24/7 like crypto)
✅ Regime detection aligned with crisis events (COVID, IL&FS, taper tantrum)
✅ Cross-sectional features (beta, correlation, relative volatility)
✅ Stock-specific fine-tuning for production deployment
✅ VaR/ES for risk management in portfolio context

This enhanced roadmap specifically addresses the unique challenges and opportunities of Indian equity markets! 🚀


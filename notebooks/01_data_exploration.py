import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.data_loader import load_all_stocks

# Load data
print("Loading data...")
data = load_all_stocks()

# 1. Summary Statistics
print("\n=== Data Summary ===")
stats = []
for symbol, df in data.items():
    stats.append({
        'Symbol': symbol,
        'Rows': len(df),
        'Start Date': df['datetime'].min(),
        'End Date': df['datetime'].max(),
        'Missing Values': df.isnull().sum().sum(),
        'Mean Volume': df['volume'].mean()
    })
    
stats_df = pd.DataFrame(stats)
print(stats_df.to_markdown())

# 2. Check for Missing Timestamps (Basic)
print("\n=== Gap Analysis (Top 5 Gaps) ===")
for symbol, df in data.items():
    df = df.sort_values('datetime')
    df['diff'] = df['datetime'].diff()
    gaps = df[df['diff'] > pd.Timedelta(minutes=5)].sort_values('diff', ascending=False).head(3)
    if not gaps.empty:
        print(f"\n{symbol} Gaps:")
        print(gaps[['datetime', 'diff']])

# 3. Correlation Matrix (Closing Prices)
print("\n=== Correlation Matrix ===")
closes = pd.DataFrame()
for symbol, df in data.items():
    closes[symbol] = df.set_index('datetime')['close']

corr = closes.corr()
print(corr)

# Generate Plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Stock Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("\nSaved correlation_matrix.png")

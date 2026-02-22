from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Stock List
STOCKS = [
    'ADANIPORTS', 'AXISBANK', 'COALINDIA', 'HAL', 'HDFCBANK',
    'ICICIBANK', 'INFY', 'LT', 'RELIANCE', 'SBIN',
    'TATASTEEL', 'TCS', 'TITAN', 'WIPRO'
]

# Data Schema
COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

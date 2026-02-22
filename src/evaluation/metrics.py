import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def qlike_loss(y_true, y_pred):
    """
    QLIKE Loss (Quasi-Likelihood).
    Loss = log(y_pred) + y_true / y_pred
    Standard metric for volatility forecasting.
    """
    # Avoid div by zero
    y_pred = np.maximum(y_pred, 1e-6)
    return np.mean(np.log(y_pred) + y_true / y_pred)

def evaluate_forecasts(y_true, y_pred):
    """
    Compute standard metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    qlike = qlike_loss(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'QLIKE': qlike
    }

def directional_accuracy(y_true, y_pred):
    """
    Direction Accuracy (DA).
    Checks if the direction of change is correctly predicted.
    """
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    
    # Sign of change
    correct_direction = np.sign(diff_true) == np.sign(diff_pred)
    return np.mean(correct_direction)

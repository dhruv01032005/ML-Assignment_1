import numpy as np
import pandas as pd

# Calculating Accuracy
def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    return (y_hat == y).mean()

# Calculating Precision
def precision(y_hat: pd.Series, y: pd.Series, cls) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

# Calculating Recall
def recall(y_hat: pd.Series, y: pd.Series, cls) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

# Calculating Mean Squared Error
def mse(y_hat: pd.Series, y: pd.Series) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    return np.mean((y_hat - y) ** 2)

# Calculating Root Mean Squared Error
def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    return np.sqrt(((y_hat - y) ** 2).mean())

# Calculating Mean Absolute Error
def mae(y_hat: pd.Series, y: pd.Series) -> float:
    if y_hat.size != y.size:
        raise ValueError("Predictions and labels must be of same size")
    return np.abs(y_hat - y).mean()

# This condition is used to validate whether the size of the series matches or not
# if y_hat.size != y.size:
#         raise ValueError("Predictions and labels must be of same size")
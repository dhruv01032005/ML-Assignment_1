import numpy as np
import pandas as pd

# Function to one hot encode if the data is string discrete values
def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)

# Check if the y is real or discrete
def check_ifreal(y: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(y)

# Calculating the entropy
def entropy(Y: pd.Series) -> float:
    probs = Y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))

# Calculating the gini_index
def gini_index(Y: pd.Series) -> float:
    probs = Y.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

# Calculating the mse
def mse(Y: pd.Series) -> float:
    if len(Y) == 0:
        return 0
    return np.mean((Y - np.mean(Y)) ** 2)

# Finding the best information gain for a specific column feature
def information_gain(Y: pd.Series, X_col: pd.Series, criterion: str, is_regression: bool):
    if is_regression:
        base = mse(Y)
    else:
        base = entropy(Y) if criterion == "information_gain" else gini_index(Y)

    best_gain, best_thresh = -1, None
    values = np.unique(X_col)

    if pd.api.types.is_numeric_dtype(X_col):
        thresholds = (values[:-1] + values[1:]) / 2
        for t in thresholds:
            left = Y[X_col <= t]
            right = Y[X_col > t]
            if len(left) == 0 or len(right) == 0:
                continue
            if is_regression:
                gain = base - (len(left)/len(Y))*mse(left) - (len(right)/len(Y))*mse(right)
            else:
                if criterion == "information_gain":
                    gain = base - (len(left)/len(Y))*entropy(left) - (len(right)/len(Y))*entropy(right)
                else:
                    gain = base - (len(left)/len(Y))*gini_index(left) - (len(right)/len(Y))*gini_index(right)
            if gain > best_gain:
                best_gain, best_thresh = gain, t
    else:
        for v in values:
            left = Y[X_col == v]
            right = Y[X_col != v]
            if len(left) == 0 or len(right) == 0:
                continue
            if is_regression:
                gain = base - (len(left)/len(Y))*mse(left) - (len(right)/len(Y))*mse(right)
            else:
                if criterion == "information_gain":
                    gain = base - (len(left)/len(Y))*entropy(left) - (len(right)/len(Y))*entropy(right)
                else:
                    gain = base - (len(left)/len(Y))*gini_index(left) - (len(right)/len(Y))*gini_index(right)
            if gain > best_gain:
                best_gain, best_thresh = gain, v
    return best_gain, best_thresh

# Finding the best information gain attribute from the whole X dataset
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features, is_regression):
    best_gain, best_attr, best_thresh = -1, None, None
    for f in features:
        gain, thresh = information_gain(y, X[f], criterion, is_regression)
        if gain > best_gain:
            best_gain, best_attr, best_thresh = gain, f, thresh
    return best_attr, best_thresh

# Spliting the X dataset based on the best information gain attribute and its threashold
def split_data(X: pd.DataFrame, y: pd.Series, attribute, threshold):
    if pd.api.types.is_numeric_dtype(X[attribute]):
        mask = X[attribute] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]
    else:
        mask = X[attribute] == threshold
        return X[mask], y[mask], X[~mask], y[~mask]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# -------------------------------
# Load dataset
# -------------------------------
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight","acceleration", "model year", "origin", "car name"])

# Cleaning
data = data.replace("?", np.nan)
data = data.dropna()
data = data.drop(columns=["car name"])
data["horsepower"] = data["horsepower"].astype(float)

# Features and target (keep X as DataFrame)
X = data.drop(columns=["mpg"])
y = data["mpg"].values

# Manual train/test split (70/30)
n = len(X)
indices = np.arange(n)
np.random.shuffle(indices)

split = int(0.7 * n)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -------------------------------
# Custom Decision Tree
# -------------------------------
custom_tree = DecisionTree(max_depth=10)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)

# -------------------------------
# Sklearn DecisionTreeRegressor
# -------------------------------
sk_tree = DecisionTreeRegressor(criterion='squared_error',max_depth=10)
sk_tree.fit(X_train, y_train)
y_pred_sklearn = sk_tree.predict(X_test)

# -------------------------------
# Evaluate metrics
# -------------------------------
mse_custom = mse(y_test, y_pred_custom)
mse_sklearn = mse(y_test, y_pred_sklearn)

print("Custom Decision Tree Performance:")
print("MSE:", mse_custom)

print("\nSklearn Decision Tree Performance:")
print("MSE:", mse_sklearn)

# -------------------------------
# Comparison table
# -------------------------------
comparison = pd.DataFrame({
    "True MPG": y_test,
    "Custom Tree Predicted MPG": y_pred_custom,
    "Sklearn Tree Predicted MPG": y_pred_sklearn
})
comparison.index.name = "Index"
print("\nSample Predictions:\n", comparison.head(10))

# -------------------------------
# Plot Custom predictions vs true
# Plot Sklearn predictions vs true
# -------------------------------
plt.scatter(y_test, y_pred_custom, alpha=0.6, label="Custom Tree")
plt.scatter(y_test, y_pred_sklearn, alpha=0.6, label="Sklearn Tree")
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.legend()
plt.title("True vs Predicted MPG")
plt.show()

# # Printing Results
# python -u "c:\Users\dhruv\Desktop\ML\assignment1\es335-25-fall-assignment-1-master\auto-efficiency.py"
# c:\Users\dhruv\Desktop\ML\assignment 1\es335-25-fall-assignment-1-master\auto-efficiency.py:14: FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead   
#   data = pd.read_csv(url, delim_whitespace=True, header=None,
# Custom Decision Tree Performance:
# MSE: 12.596367375496968

# Sklearn Decision Tree Performance:
# MSE: 13.763147036513914

# Sample Predictions:
#         True MPG  Custom Tree Predicted MPG  Sklearn Tree Predicted MPG
# Index
# 4          17.0                  16.000000                   18.000000
# 258        20.6                  20.500000                   19.400000
# 355        33.7                  32.400000                   32.400000
# 101        23.0                  22.000000                   22.000000
# 228        18.5                  17.500000                   17.500000
# 339        26.6                  27.000000                   25.800000
# 215        13.0                  13.444444                   13.444444
# 173        24.0                  24.300000                   24.300000
# 99         18.0                  19.500000                   20.000000
# 217        30.0                  25.000000                   24.000000
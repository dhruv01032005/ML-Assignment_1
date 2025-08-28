import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Dataset generation
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Dataset")
plt.show()

# Convert to DataFrame/Series
X = pd.DataFrame(X, columns=["f1", "f2"])
y = pd.Series(y, dtype="category")

# -------------------------------
# Q2 (a) Train/Test split (70/30)
# -------------------------------
split = int(0.7 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("\n--- Q2 (a): Train/Test Evaluation ---")
print("Accuracy:", accuracy(y_hat, y_test))
for cls in y_test.unique():
    print(f"Class {cls} -> Precision: {precision(y_hat, y_test, cls):.3f}, Recall: {recall(y_hat, y_test, cls):.3f}")

# -------------------------------
# Q2 (b) Nested 5-fold Cross Validation
# -------------------------------
from sklearn.model_selection import KFold

print("\n--- Q2 (b): Nested Cross-Validation ---")

kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
outer_fold_results = []
best_depths = []
for train_index, test_index in kf_outer.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    inner_scores = {}
    for depth in range(1, 8):
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for itrain_idx, ival_idx in inner_kf.split(X_train):
            Xi_train, Xi_val = X_train.iloc[itrain_idx], X_train.iloc[ival_idx]
            yi_train, yi_val = y_train.iloc[itrain_idx], y_train.iloc[ival_idx]

            model = DecisionTree(criterion="information_gain", max_depth=depth)
            model.fit(Xi_train, yi_train)
            yi_hat = model.predict(Xi_val)
            scores.append(accuracy(yi_hat, yi_val))

        inner_scores[depth] = np.mean(scores)

    best_depth = max(inner_scores, key=inner_scores.get)
    best_depths.append(best_depth)

    # Retrain on full train set with best depth
    best_model = DecisionTree(criterion="information_gain", max_depth=best_depth)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    fold_acc = accuracy(y_pred, y_test)
    outer_fold_results.append(float(fold_acc))

# Printing the results
print("Best depths per outer fold:", best_depths)
vals, counts = np.unique(best_depth, return_counts=True)
print("Optimal depth:", vals[np.argmax(counts)])
print("Outer CV accuracies:", outer_fold_results)
print("Mean Outer CV Accuracy:", np.mean(np.array(outer_fold_results)))

# # Printed Results
# python -u "c:\Users\dhruv\Desktop\ML\assignment1\es335-25-fall-assignment-1-master\classification-exp.py"

# --- Q2 (a): Train/Test Evaluation ---
# Accuracy: 0.9
# Class 0 -> Precision: 0.909, Recall: 0.833
# Class 1 -> Precision: 0.895, Recall: 0.944

# --- Q2 (b): Nested Cross-Validation ---
# Best depths per outer fold: [1, 1, 1, 2, 1]
# Optimal depth: 1
# Outer CV accuracies: [0.95, 0.8, 0.85, 0.95, 0.95]
# Mean Outer CV Accuracy: 0.9
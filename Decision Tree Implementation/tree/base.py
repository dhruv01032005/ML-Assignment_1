from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)

# Nodes for storing the decision tree which contains the feature, threshold, left node, right node and predicted value for leaf node. 
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree Class
@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    # Initializing the parameters
    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = None
        
    # Function to train decision tree
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.is_regression = check_ifreal(y)
        self.root = self._build_tree(X, y, depth=0)

    # Helps in building tree
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or X.shape[0] == 0:
            return Node(value=self._leaf_value(y))

        feature, threshold = opt_split_attribute(X, y, self.criterion, X.columns, self.is_regression)
        if feature is None:
            return Node(value=self._leaf_value(y))
        X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)
    
    # When the tree either reaches the maximum depth or all the labels are same so it makes the leaf node
    def _leaf_value(self, y):
        if len(y) == 0:
            return None
        if self.is_regression:
            return np.mean(y)
        else:
            return y.mode()[0]

    # Function to predict the value from the decision tree made
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda row: self._traverse(row, self.root), axis=1)

    # Traversing into the branches of the tree based on the conditions specified
    def _traverse(self, row, node: Node):
        if node.value is not None:
            return node.value
        if pd.api.types.is_numeric_dtype(row[node.feature]):
            if row[node.feature] <= node.threshold:
                return self._traverse(row, node.left)
            else:
                return self._traverse(row, node.right)
        else:
            if row[node.feature] == node.threshold:
                return self._traverse(row, node.left)
            else:
                return self._traverse(row, node.right)
            
    # Printing the whole tree made in the console
    def plot(self) -> None:
        def recurse(node, depth=0):
            indent = "    " * depth
            if node.value is not None:
                print(f"Predict -> {node.value}")
            else:
                print(f"?(X[{node.feature}] <= {node.threshold})")
                print(f"{indent}    Y:",end=" ")
                recurse(node.left, depth + 1)
                print(f"{indent}    N:",end=" ")
                recurse(node.right, depth + 1)
        recurse(self.root)
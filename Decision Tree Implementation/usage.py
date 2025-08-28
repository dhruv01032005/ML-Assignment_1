import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

print("Real Input and Real Output")
tree = DecisionTree()  # Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))
print('--------------------------------------------------')
print('\n')

# Test case 2
# Real Input and Discrete Output

N = 30
P = 5
M = 3

X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(M, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print("Real Input and Discrete Output")
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    print("Criteria :", criteria)
    tree.plot()
    print("Accuracy: ", accuracy(y_hat, y))
    print('--------------------------------------------------')
    for cls in y.unique():
        print("Class = ",cls)
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
        print('--------------------------------------------------')
    print('\n')

# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
M = 3
M_1 = 10

X = pd.DataFrame({i: pd.Series(np.random.randint(M_1, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randint(M, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print("Discrete Input and Discrete Output")
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    print("Criteria :", criteria)
    tree.plot()
    print("Accuracy: ", accuracy(y_hat, y))
    print('--------------------------------------------------')
    for cls in y.unique():
        print("Class = ",cls)
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
        print('--------------------------------------------------')
    print('\n')
    
# Test case 4
# Discrete Input and Real Output

N = 30
M_1 = 10
P  = 5

X = pd.DataFrame({i: pd.Series(np.random.randint(M_1, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randn(N))

print("Discrete Input and Real Output")
tree = DecisionTree()
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))
print('--------------------------------------------------')
print('\n')

# # Printing Results
# python -u "c:\Users\dhruv\Desktop\ML\assignment1\es335-25-fall-assignment-1-master\usage.py"
# Real Input and Real Output
# ?(X[1] <= -1.1939721441501923)
#     Y: Predict -> 2.720169166589619
#     N: ?(X[4] <= 0.3865751136808898)
#         Y: ?(X[0] <= 1.5077915869695468)
#             Y: ?(X[4] <= -1.7439789939378834)
#                 Y: ?(X[1] <= 0.8125418173520702)
#                     Y: Predict -> 0.534667267785164
#                     N: Predict -> 0.9633761292443218
#                 N: ?(X[2] <= 1.24071347131677)
#                     Y: Predict -> -0.345994377277823
#                     N: Predict -> 0.82206015999449
#             N: ?(X[0] <= 1.870195015413759)
#                 Y: Predict -> 1.4535340771573169
#                 N: Predict -> 0.8271832490360238
#         N: ?(X[0] <= -0.35665559739723524)
#             Y: ?(X[0] <= -0.4904656407149133)
#                 Y: ?(X[0] <= -0.5517318279069667)
#                     Y: Predict -> 1.8657745111447566
#                     N: Predict -> 1.8967929826539474
#                 N: Predict -> 1.158595579007404
#             N: ?(X[4] <= 1.003272324809155)
#                 Y: ?(X[0] <= 0.8611560330796227)
#                     Y: Predict -> 0.3214303278812129
#                     N: Predict -> 0.787084603742452
#                 N: Predict -> -0.9746816702273214
# RMSE:  0.3601329774475513
# MAE:  0.24067253750418452
# --------------------------------------------------


# Real Input and Discrete Output
# Criteria : information_gain
# ?(X[3] <= 0.8315558665108316)
#     Y: ?(X[0] <= -0.9323777557466029)
#         Y: Predict -> 1
#         N: ?(X[4] <= 0.654613295429144)
#             Y: ?(X[1] <= 0.5775952979342543)
#                 Y: ?(X[1] <= -0.6665031273672707)
#                     Y: Predict -> 1
#                     N: Predict -> 0
#                 N: ?(X[0] <= 0.3441089470009745)
#                     Y: Predict -> 1
#                     N: Predict -> 2
#             N: ?(X[2] <= 0.11601307733662071)
#                 Y: Predict -> 2
#                 N: Predict -> 0
#     N: ?(X[0] <= -1.026702346383873)
#         Y: Predict -> 1
#         N: Predict -> 2
# Accuracy:  0.9333333333333333
# --------------------------------------------------
# Class =  0
# Precision:  1.0
# Recall:  0.7777777777777778
# --------------------------------------------------
# Class =  1
# Precision:  0.8571428571428571
# Recall:  1.0
# --------------------------------------------------
# Class =  2
# Precision:  1.0
# Recall:  1.0
# --------------------------------------------------


# Real Input and Discrete Output
# Criteria : gini_index
# ?(X[0] <= -0.9323777557466029)
#     Y: Predict -> 1
#     N: ?(X[3] <= 0.8315558665108316)
#         Y: ?(X[4] <= 0.654613295429144)
#             Y: ?(X[1] <= 0.5775952979342543)
#                 Y: ?(X[1] <= -0.6665031273672707)
#                     Y: Predict -> 1
#                     N: Predict -> 0
#                 N: ?(X[0] <= 0.3441089470009745)
#                     Y: Predict -> 1
#                     N: Predict -> 2
#             N: ?(X[2] <= 0.11601307733662071)
#                 Y: Predict -> 2
#                 N: Predict -> 0
#         N: Predict -> 2
# Accuracy:  0.9333333333333333
# --------------------------------------------------
# Class =  0
# Precision:  1.0
# Recall:  0.7777777777777778
# --------------------------------------------------
# Class =  1
# Precision:  0.8571428571428571
# Recall:  1.0
# --------------------------------------------------
# Class =  2
# Precision:  1.0
# Recall:  1.0
# --------------------------------------------------


# Discrete Input and Discrete Output
# Criteria : information_gain
# ?(X[3] <= 4)
#     Y: ?(X[1] <= 4)
#         Y: Predict -> 2
#         N: ?(X[1] <= 2)
#             Y: Predict -> 0
#             N: ?(X[1] <= 8)
#                 Y: Predict -> 0
#                 N: Predict -> 1
#     N: ?(X[1] <= 6)
#         Y: Predict -> 0
#         N: ?(X[2] <= 2)
#             Y: Predict -> 0
#             N: ?(X[0] <= 2)
#                 Y: Predict -> 2
#                 N: ?(X[3] <= 2)
#                     Y: Predict -> 2
#                     N: Predict -> 1
# Accuracy:  0.43333333333333335
# --------------------------------------------------
# Class =  0
# Precision:  0.2857142857142857
# Recall:  0.6666666666666666
# --------------------------------------------------
# Class =  1
# Precision:  1.0
# Recall:  0.14285714285714285
# --------------------------------------------------
# Class =  2
# Precision:  0.5
# Recall:  0.7
# --------------------------------------------------


# Discrete Input and Discrete Output
# Criteria : gini_index
# ?(X[1] <= 2)
#     Y: ?(X[3] <= 4)
#         Y: Predict -> 0
#         N: Predict -> 2
#     N: ?(X[1] <= 8)
#         Y: ?(X[0] <= 4)
#             Y: Predict -> 1
#             N: ?(X[0] <= 5)
#                 Y: Predict -> 2
#                 N: Predict -> 0
#         N: ?(X[2] <= 5)
#             Y: Predict -> 2
#             N: ?(X[1] <= 6)
#                 Y: Predict -> 0
#                 N: ?(X[0] <= 1)
#                     Y: Predict -> 2
#                     N: Predict -> 1
# Accuracy:  0.4
# --------------------------------------------------
# Class =  0
# Precision:  0.38461538461538464
# Recall:  0.8333333333333334
# --------------------------------------------------
# Class =  1
# Precision:  0.4444444444444444
# Recall:  0.2857142857142857
# --------------------------------------------------
# Class =  2
# Precision:  0.375
# Recall:  0.3
# --------------------------------------------------


# Discrete Input and Real Output
# ?(X[4] <= 8)
#     Y: ?(X[1] <= 0)
#         Y: Predict -> -2.4716445001272893
#         N: Predict -> -1.6275424378831627
#     N: ?(X[1] <= 7)
#         Y: ?(X[2] <= 9)
#             Y: Predict -> 0.37114587337130883
#             N: ?(X[0] <= 2)
#                 Y: Predict -> 1.1677820616598074
#                 N: Predict -> 1.4415686206579004
#         N: ?(X[1] <= 0)
#             Y: ?(X[0] <= 7)
#                 Y: Predict -> -1.6615200622689599
#                 N: ?(X[0] <= 8)
#                     Y: Predict -> -0.2030453860429927
#                     N: Predict -> -0.7321992759998865
#             N: ?(X[0] <= 5)
#                 Y: ?(X[1] <= 8)
#                     Y: Predict -> -1.129706854657618
#                     N: Predict -> -0.6518361078021592
#                 N: ?(X[0] <= 8)
#                     Y: Predict -> 1.0062928092144405
#                     N: Predict -> -0.04957384476473601
# RMSE:  1.6822431071331132
# MAE:  1.4764033252558024
# --------------------------------------------------
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 20  # Number of times to run each experiment to calculate the average values

def plot_results(N,P,fit,predict):
    n = len(N)
    m = len(P)
    fit_2 = fit.T
    predict_2 = predict.T
    for i in range(m):
        plt.subplot(2,m,i+1)
        plt.plot(N,fit_2[i])
        plt.title(f"Fiting time for P={P[i]}")
    for i in range(m):
        plt.subplot(2,m,m+i+1)
        plt.plot(N,predict_2[i])
        plt.title(f"Predicting time for P={P[i]}")
    plt.tight_layout()
    plt.show()
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.plot(P,fit[i])
        plt.title(f"Fiting time for N={N[i]}")
    for i in range(n):
        plt.subplot(2,n,n+i+1)
        plt.plot(P,predict[i])
        plt.ylim(np.min(predict[i])-0.1,np.max(predict[i])+0.1)
        plt.title(f"Predicting time for N={N[i]}")
    plt.tight_layout()
    plt.show()

N = [30,50,100]
P = [2,5,10]

# Test case 1
# Real Input and Real Output
print("Real Input and Real Output")
time_taken_fit_1 = np.zeros((len(N),len(P)))
time_taken_predict_1 = np.zeros((len(N),len(P)))

for i in range(len(N)):
    for j in range(len(P)):    
        X = pd.DataFrame(np.random.randn(N[i], P[j]))
        y = pd.Series(np.random.randn(N[i]))
        time_taken_fit = np.zeros(num_average_time)
        time_taken_predict = np.zeros(num_average_time)
        for k in range(num_average_time):
            tree = DecisionTree(max_depth=3)
            start_time = time.time()
            tree.fit(X, y)
            end_time = time.time()
            time_taken_fit[k] = (end_time-start_time)/num_average_time
            start_time = time.time()
            tree.predict(X)
            end_time = time.time()
            time_taken_predict[k] = (end_time-start_time)/num_average_time
        
        avg_time_taken_fit = np.mean(time_taken_fit)
        std_time_taken_fit = np.std(time_taken_fit)
        avg_time_taken_predict = np.mean(time_taken_predict)
        std_time_taken_predict = np.std(time_taken_predict)
        time_taken_fit_1[i][j] = avg_time_taken_fit
        time_taken_predict_1[i][j] = avg_time_taken_predict
        print(f"Time taken to fit for N = {N[i]}, P = {P[j]} = {avg_time_taken_fit}")
        print(f"Standard Deviation of Time taken to fit for N = {N[i]}, P = {P[j]} = {std_time_taken_fit}")
        print(f"Time taken to predict for N = {N[i]}, P = {P[j]} = {avg_time_taken_predict}")
        print(f"Standard Deviation of Time taken to predict for N = {N[i]}, P = {P[j]} = {std_time_taken_predict}")


# Test case 2
# Real Input and Discrete Output
print("\nReal Input and Discrete Output")
time_taken_fit_2 = np.zeros((len(N),len(P)))
time_taken_predict_2 = np.zeros((len(N),len(P)))

for i in range(len(N)):
    for j in range(len(P)):
        X = pd.DataFrame(np.random.randn(N[i], P[j]))
        y = pd.Series(np.random.randint(P[j], size=N[i]), dtype="category")
        time_taken_fit = np.zeros(num_average_time)
        time_taken_predict = np.zeros(num_average_time)
        for k in range(num_average_time):
            tree = DecisionTree(max_depth=3)
            start_time = time.time()
            tree.fit(X, y)
            end_time = time.time()
            time_taken_fit[k] = (end_time-start_time)/num_average_time
            start_time = time.time()
            tree.predict(X)
            end_time = time.time()
            time_taken_predict[k] = (end_time-start_time)/num_average_time
        
        avg_time_taken_fit = np.mean(time_taken_fit)
        std_time_taken_fit = np.std(time_taken_fit)
        avg_time_taken_predict = np.mean(time_taken_predict)
        std_time_taken_predict = np.std(time_taken_predict)
        time_taken_fit_2[i][j] = avg_time_taken_fit
        time_taken_predict_2[i][j] = avg_time_taken_predict
        print(f"Time taken to fit for N = {N[i]}, P = {P[j]} = {avg_time_taken_fit}")
        print(f"Standard Deviation of Time taken to fit for N = {N[i]}, P = {P[j]} = {std_time_taken_fit}")
        print(f"Time taken to predict for N = {N[i]}, P = {P[j]} = {avg_time_taken_predict}")
        print(f"Standard Deviation of Time taken to predict for N = {N[i]}, P = {P[j]} = {std_time_taken_predict}")

# Test case 3
# Discrete Input and Discrete Output
print("\nDiscrete Input and Discrete Output")
time_taken_fit_3 = np.zeros((len(N),len(P)))
time_taken_predict_3 = np.zeros((len(N),len(P)))

for i in range(len(N)):
    for j in range(len(P)):
        X = pd.DataFrame({k: pd.Series(np.random.randint(2, size=N[i]), dtype="category") for k in range(P[j])})
        y = pd.Series(np.random.randint(P[j], size=N[i]), dtype="category")
        time_taken_fit = np.zeros(num_average_time)
        time_taken_predict = np.zeros(num_average_time)
        for k in range(num_average_time):
            tree = DecisionTree(max_depth=3)
            start_time = time.time()
            tree.fit(X, y)
            end_time = time.time()
            time_taken_fit[k] = (end_time-start_time)/num_average_time
            start_time = time.time()
            tree.predict(X)
            end_time = time.time()
            time_taken_predict[k] = (end_time-start_time)/num_average_time
        
        avg_time_taken_fit = np.mean(time_taken_fit)
        std_time_taken_fit = np.std(time_taken_fit)
        avg_time_taken_predict = np.mean(time_taken_predict)
        std_time_taken_predict = np.std(time_taken_predict)
        time_taken_fit_3[i][j] = avg_time_taken_fit
        time_taken_predict_3[i][j] = avg_time_taken_predict
        print(f"Time taken to fit for N = {N[i]}, P = {P[j]} = {avg_time_taken_fit}")
        print(f"Standard Deviation of Time taken to fit for N = {N[i]}, P = {P[j]} = {std_time_taken_fit}")
        print(f"Time taken to predict for N = {N[i]}, P = {P[j]} = {avg_time_taken_predict}")
        print(f"Standard Deviation of Time taken to predict for N = {N[i]}, P = {P[j]} = {std_time_taken_predict}")

# Test case 4
# Discrete Input and Real Output
print("\nDiscrete Input and Real Output")
time_taken_fit_4 = np.zeros((len(N),len(P)))
time_taken_predict_4 = np.zeros((len(N),len(P)))

for i in range(len(N)):
    for j in range(len(P)):
        X = pd.DataFrame({k: pd.Series(np.random.randint(2, size=N[i]), dtype="category") for k in range(P[j])})
        y = pd.Series(np.random.randn(N[i]))
        time_taken_fit = np.zeros(num_average_time)
        time_taken_predict = np.zeros(num_average_time)
        for k in range(num_average_time):
            tree = DecisionTree(max_depth=3)
            start_time = time.time()
            tree.fit(X, y)
            end_time = time.time()
            time_taken_fit[k] = (end_time-start_time)/num_average_time
            start_time = time.time()
            tree.predict(X)
            end_time = time.time()
            time_taken_predict[k] = (end_time-start_time)/num_average_time
        
        avg_time_taken_fit = np.mean(time_taken_fit)
        std_time_taken_fit = np.std(time_taken_fit)
        avg_time_taken_predict = np.mean(time_taken_predict)
        std_time_taken_predict = np.std(time_taken_predict)
        time_taken_fit_4[i][j] = avg_time_taken_fit
        time_taken_predict_4[i][j] = avg_time_taken_predict
        print(f"Time taken to fit for N = {N[i]}, P = {P[j]} = {avg_time_taken_fit}")
        print(f"Standard Deviation of Time taken to fit for N = {N[i]}, P = {P[j]} = {std_time_taken_fit}")
        print(f"Time taken to predict for N = {N[i]}, P = {P[j]} = {avg_time_taken_predict}")
        print(f"Standard Deviation of Time taken to predict for N = {N[i]}, P = {P[j]} = {std_time_taken_predict}")

# All the plots
plot_results(N,P,time_taken_fit_1,time_taken_predict_1)
plot_results(N,P,time_taken_fit_2,time_taken_predict_2)
plot_results(N,P,time_taken_fit_3,time_taken_predict_3)
plot_results(N,P,time_taken_fit_4,time_taken_predict_4)

# # Printed Results
# python -u "c:\Users\dhruv\Desktop\ML\assignment1\es335-25-fall-assignment-1-master\experiments.py"
# Real Input and Real Output
# Time taken to fit for N = 30, P = 2 = 0.0033701515197753903
# Standard Deviation of Time taken to fit for N = 30, P = 2 = 0.00048240382926390175
# Time taken to predict for N = 30, P = 2 = 3.678143024444581e-05
# Standard Deviation of Time taken to predict for N = 30, P = 2 = 2.1619593274216877e-05
# Time taken to fit for N = 30, P = 5 = 0.008259246945381165
# Standard Deviation of Time taken to fit for N = 30, P = 5 = 0.000954808412811009
# Time taken to predict for N = 30, P = 5 = 2.8394460678100586e-05
# Standard Deviation of Time taken to predict for N = 30, P = 5 = 4.273904742458891e-06
# Time taken to fit for N = 30, P = 10 = 0.015158900022506713
# Standard Deviation of Time taken to fit for N = 30, P = 10 = 0.0009977208066137132
# Time taken to predict for N = 30, P = 10 = 3.058791160583496e-05
# Standard Deviation of Time taken to predict for N = 30, P = 10 = 9.670281448395906e-06
# Time taken to fit for N = 50, P = 2 = 0.0055399018526077265
# Standard Deviation of Time taken to fit for N = 50, P = 2 = 0.0003370817335763905
# Time taken to predict for N = 50, P = 2 = 3.8342475891113286e-05
# Standard Deviation of Time taken to predict for N = 50, P = 2 = 3.666204904205628e-06
# Time taken to fit for N = 50, P = 5 = 0.013023037910461426
# Standard Deviation of Time taken to fit for N = 50, P = 5 = 0.0007352641115427949
# Time taken to predict for N = 50, P = 5 = 4.3358206748962406e-05
# Standard Deviation of Time taken to predict for N = 50, P = 5 = 1.3313269907091451e-05
# Time taken to fit for N = 50, P = 10 = 0.025469893217086793
# Standard Deviation of Time taken to fit for N = 50, P = 10 = 0.001057729442443206
# Time taken to predict for N = 50, P = 10 = 4.027009010314941e-05
# Standard Deviation of Time taken to predict for N = 50, P = 10 = 5.453403383645086e-06
# Time taken to fit for N = 100, P = 2 = 0.010874226093292236
# Standard Deviation of Time taken to fit for N = 100, P = 2 = 0.0006361472321669919
# Time taken to predict for N = 100, P = 2 = 7.407307624816895e-05
# Standard Deviation of Time taken to predict for N = 100, P = 2 = 1.5542309996862047e-05
# Time taken to fit for N = 100, P = 5 = 0.025991981029510496
# Standard Deviation of Time taken to fit for N = 100, P = 5 = 0.0008081124061237918
# Time taken to predict for N = 100, P = 5 = 7.074594497680664e-05
# Standard Deviation of Time taken to predict for N = 100, P = 5 = 9.151839628100729e-06
# Time taken to fit for N = 100, P = 10 = 0.054100772738456725
# Standard Deviation of Time taken to fit for N = 100, P = 10 = 0.0037921513263970003
# Time taken to predict for N = 100, P = 10 = 7.78275728225708e-05
# Standard Deviation of Time taken to predict for N = 100, P = 10 = 1.8364395259559433e-05

# Real Input and Discrete Output
# Time taken to fit for N = 30, P = 2 = 0.006172041296958924
# Standard Deviation of Time taken to fit for N = 30, P = 2 = 0.0005714649775216369
# Time taken to predict for N = 30, P = 2 = 2.9208064079284663e-05
# Standard Deviation of Time taken to predict for N = 30, P = 2 = 1.1578267697477009e-05
# Time taken to fit for N = 30, P = 5 = 0.01860660433769226
# Standard Deviation of Time taken to fit for N = 30, P = 5 = 0.0007009199705738747
# Time taken to predict for N = 30, P = 5 = 2.703309059143067e-05
# Standard Deviation of Time taken to predict for N = 30, P = 5 = 3.541703980121632e-06
# Time taken to fit for N = 30, P = 10 = 0.03606779158115388
# Standard Deviation of Time taken to fit for N = 30, P = 10 = 0.001270296969469084
# Time taken to predict for N = 30, P = 10 = 2.7394294738769533e-05
# Standard Deviation of Time taken to predict for N = 30, P = 10 = 2.8351693474187242e-06
# Time taken to fit for N = 50, P = 2 = 0.012182355523109435
# Standard Deviation of Time taken to fit for N = 50, P = 2 = 0.0007893595689292758
# Time taken to predict for N = 50, P = 2 = 4.170656204223633e-05
# Standard Deviation of Time taken to predict for N = 50, P = 2 = 9.384570952111954e-06
# Time taken to fit for N = 50, P = 5 = 0.02908993601799011
# Standard Deviation of Time taken to fit for N = 50, P = 5 = 0.0012633313701506713
# Time taken to predict for N = 50, P = 5 = 4.0951371192932126e-05
# Standard Deviation of Time taken to predict for N = 50, P = 5 = 1.0160388158734805e-05
# Time taken to fit for N = 50, P = 10 = 0.061783559322357164
# Standard Deviation of Time taken to fit for N = 50, P = 10 = 0.002475526480329657
# Time taken to predict for N = 50, P = 10 = 4.237174987792969e-05
# Standard Deviation of Time taken to predict for N = 50, P = 10 = 6.473665271011242e-06
# Time taken to fit for N = 100, P = 2 = 0.023932671546936034
# Standard Deviation of Time taken to fit for N = 100, P = 2 = 0.0011050811203518588
# Time taken to predict for N = 100, P = 2 = 7.681071758270264e-05
# Standard Deviation of Time taken to predict for N = 100, P = 2 = 1.9861244428293752e-05
# Time taken to fit for N = 100, P = 5 = 0.06165869593620301
# Standard Deviation of Time taken to fit for N = 100, P = 5 = 0.0012707640594919677
# Time taken to predict for N = 100, P = 5 = 8.18425416946411e-05
# Standard Deviation of Time taken to predict for N = 100, P = 5 = 2.1209438093413332e-05
# Time taken to fit for N = 100, P = 10 = 0.1226427286863327
# Standard Deviation of Time taken to fit for N = 100, P = 10 = 0.0021153460355074747
# Time taken to predict for N = 100, P = 10 = 7.191359996795654e-05
# Standard Deviation of Time taken to predict for N = 100, P = 10 = 9.237896571078182e-06

# Discrete Input and Discrete Output
# Time taken to fit for N = 30, P = 2 = 0.0009547716379165649
# Standard Deviation of Time taken to fit for N = 30, P = 2 = 0.00010981355764623274
# Time taken to predict for N = 30, P = 2 = 0.00029624342918396
# Standard Deviation of Time taken to predict for N = 30, P = 2 = 7.003985453053414e-05
# Time taken to fit for N = 30, P = 5 = 0.00325007677078247
# Standard Deviation of Time taken to fit for N = 30, P = 5 = 0.0004211896362648775
# Time taken to predict for N = 30, P = 5 = 0.0003204488754272461
# Standard Deviation of Time taken to predict for N = 30, P = 5 = 5.550092169388465e-05
# Time taken to fit for N = 30, P = 10 = 0.006825221180915833
# Standard Deviation of Time taken to fit for N = 30, P = 10 = 0.0016144303181220845
# Time taken to predict for N = 30, P = 10 = 0.00034882545471191405
# Standard Deviation of Time taken to predict for N = 30, P = 10 = 5.435543105676715e-05
# Time taken to fit for N = 50, P = 2 = 0.001577383279800415
# Standard Deviation of Time taken to fit for N = 50, P = 2 = 0.0003667728259998942
# Time taken to predict for N = 50, P = 2 = 0.0008263671398162841
# Standard Deviation of Time taken to predict for N = 50, P = 2 = 0.00025611519168788145
# Time taken to fit for N = 50, P = 5 = 0.006221490502357483
# Standard Deviation of Time taken to fit for N = 50, P = 5 = 0.001542599047536384
# Time taken to predict for N = 50, P = 5 = 0.0009364581108093262
# Standard Deviation of Time taken to predict for N = 50, P = 5 = 0.00031880400766193007
# Time taken to fit for N = 50, P = 10 = 0.012166958451271057
# Standard Deviation of Time taken to fit for N = 50, P = 10 = 0.0024771248166866157
# Time taken to predict for N = 50, P = 10 = 0.0010573875904083252
# Standard Deviation of Time taken to predict for N = 50, P = 10 = 0.0004246048908324511
# Time taken to fit for N = 100, P = 2 = 0.001654185652732849
# Standard Deviation of Time taken to fit for N = 100, P = 2 = 0.0004881764909371259
# Time taken to predict for N = 100, P = 2 = 0.0015754562616348267
# Standard Deviation of Time taken to predict for N = 100, P = 2 = 0.0005082760136947541
# Time taken to fit for N = 100, P = 5 = 0.005766180753707886
# Standard Deviation of Time taken to fit for N = 100, P = 5 = 0.0017972377197205877
# Time taken to predict for N = 100, P = 5 = 0.001582919955253601
# Standard Deviation of Time taken to predict for N = 100, P = 5 = 0.0005720046841400217
# Time taken to fit for N = 100, P = 10 = 0.012823926806449889
# Standard Deviation of Time taken to fit for N = 100, P = 10 = 0.002911097085341042
# Time taken to predict for N = 100, P = 10 = 0.0019463670253753664
# Standard Deviation of Time taken to predict for N = 100, P = 10 = 0.0006417420182606877

# Discrete Input and Real Output
# Time taken to fit for N = 30, P = 2 = 0.0005254709720611573
# Standard Deviation of Time taken to fit for N = 30, P = 2 = 9.282341182198098e-05
# Time taken to predict for N = 30, P = 2 = 0.00029391467571258543
# Standard Deviation of Time taken to predict for N = 30, P = 2 = 5.7780676028088495e-05
# Time taken to fit for N = 30, P = 5 = 0.0016001689434051512
# Standard Deviation of Time taken to fit for N = 30, P = 5 = 0.00019423988935561554
# Time taken to predict for N = 30, P = 5 = 0.00034292876720428463
# Standard Deviation of Time taken to predict for N = 30, P = 5 = 6.93661093581752e-05
# Time taken to fit for N = 30, P = 10 = 0.0033334684371948238
# Standard Deviation of Time taken to fit for N = 30, P = 10 = 0.000497481162302454
# Time taken to predict for N = 30, P = 10 = 0.0003795278072357178
# Standard Deviation of Time taken to predict for N = 30, P = 10 = 5.608252001990989e-05
# Time taken to fit for N = 50, P = 2 = 0.0005023235082626343
# Standard Deviation of Time taken to fit for N = 50, P = 2 = 0.0003203875413794027
# Time taken to predict for N = 50, P = 2 = 0.00039229512214660644
# Standard Deviation of Time taken to predict for N = 50, P = 2 = 7.205014627037936e-05
# Time taken to fit for N = 50, P = 5 = 0.001415128707885742
# Standard Deviation of Time taken to fit for N = 50, P = 5 = 0.00010482803389711693
# Time taken to predict for N = 50, P = 5 = 0.00043950080871582033
# Standard Deviation of Time taken to predict for N = 50, P = 5 = 2.8602634852437345e-05
# Time taken to fit for N = 50, P = 10 = 0.0032354158163070677
# Standard Deviation of Time taken to fit for N = 50, P = 10 = 0.0006962732082064334
# Time taken to predict for N = 50, P = 10 = 0.0006564766168594361
# Standard Deviation of Time taken to predict for N = 50, P = 10 = 0.0002834422548559666
# Time taken to fit for N = 100, P = 2 = 0.0008399051427841187
# Standard Deviation of Time taken to fit for N = 100, P = 2 = 0.0003308979607769296
# Time taken to predict for N = 100, P = 2 = 0.0014419126510620119
# Standard Deviation of Time taken to predict for N = 100, P = 2 = 0.00047493687921204396
# Time taken to fit for N = 100, P = 5 = 0.0027837705612182616
# Standard Deviation of Time taken to fit for N = 100, P = 5 = 0.0006976615096412081
# Time taken to predict for N = 100, P = 5 = 0.001606054902076721
# Standard Deviation of Time taken to predict for N = 100, P = 5 = 0.0004554512338337946
# Time taken to fit for N = 100, P = 10 = 0.005846710801124573
# Standard Deviation of Time taken to fit for N = 100, P = 10 = 0.001413331709908771
# Time taken to predict for N = 100, P = 10 = 0.0018971234560012818
# Standard Deviation of Time taken to predict for N = 100, P = 10 = 0.0005127578179228556
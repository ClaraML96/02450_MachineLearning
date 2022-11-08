import numpy as np
import random
import sklearn.linear_model as lm
import sys
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

from toolbox_02450 import rlr_validate, train_neural_net

# Costum scripts
from load_data import load_data
from regB_stat import regB_stat


t1 = time.perf_counter()

random.seed(42)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Parameters
k1 = 10
k2 = 10
lambda_arr = np.power(10., range(-1,7))
h_arr = [1,2,3,4,5,7,10,15]
nn_max_iter = 10000
d_digits = 4 # number of digits after decimal point 

data = load_data()
N,M = data.shape
M -= 1 # minus ozone levels

indexes = np.arange(N)
random.shuffle(indexes) # randomize indexes since the data is ordered by date
indexes=np.array(np.array_split(indexes, k1), dtype="object") # split into k1 many data intervals (all have 33 except the last one who has 32)

data = np.array(data)

lm_test_error = np.zeros(k1)
opt_lambda_arr = np.zeros(k1)
lmRegl_test_error = np.zeros(k1)
opt_h_arr = np.zeros(k1)
ann_test_error = np.zeros(k1)
tmp_ann_test_error = np.zeros(len(h_arr))

for i in range(k1):
    print("\nOuter fold round : ", i+1, " of ", k1)
    print("Error: ")
    train_data_index = [subval for sublist in np.append(indexes[:i],indexes[i+1:]) for subval in sublist]  # combine all indexes who are not in test
    X_train = data[train_data_index,1:]
    y_train = data[train_data_index,0] # ozone levels
    X_test = data[indexes[i],1:]
    y_test = data[indexes[i],0]  # ozone levels
    n_test = len(y_test) # number of testing points

    # Standardize the data with in respect of the training set
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train = (X_train - mu)/sigma
    X_test = (X_test - mu)/sigma

    ## ------------------ Compute Baseline - Linear regression model ------------------  ##
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    # lm_train_error[i] = sum((y_train-m.predict(X_train))**2)/len(y_train)   
    lm_test_error[i] = round(sum((y_test-m.predict(X_test))**2)/n_test,d_digits)
    print("- Linear Reg. : " + str(lm_test_error[i]))

    ## -------------- Compute Linear regression with regularization factor -------------- ##
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambda_arr, k2)
    opt_lambda_arr[i] = opt_lambda

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # not regularizing the bias term
    w = np.linalg.solve(X_train.T @ X_train + lambdaI , X_train.T @ y_train)
    lmRegl_test_error[i] = round(sum(y_test-X_test @ w)/n_test,d_digits)
    print("- Linear Reg. with regul. fact. (lambda="+str(opt_lambda)+"): " + str(lmRegl_test_error[i]))

    ## -------------------------------- Compute ANN --------------------------------------- ##
    print("- ANN tmp : ", end="")
    for j, n_hidden_units in enumerate(h_arr):
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features (every other attribute except ozone) to n_hidden_units 
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        loss_fn = torch.nn.MSELoss() #  mean-squared-error loss

        # print('Training model of type:\n\n{}\n'.format(str(model())))

        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(np.reshape(y_train,(-1,1)))
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(np.reshape(y_test,(-1,1)))
            
        # Train the net on training data
        with HiddenPrints(): # hide printouts from train function
            net, final_loss, learning_curve = train_neural_net(model,
                                                            loss_fn,
                                                            X=X_train,
                                                            y=y_train,
                                                            n_replicates=1,
                                                            max_iter=nn_max_iter)

        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2   # squared error
        mse = np.float64((sum(se).type(torch.float)/n_test).data.numpy()) # means squared error
        tmp_ann_test_error[j] = round(mse, d_digits)

        print(tmp_ann_test_error[j],end = " , ")

    
    index_h_opt = np.argmin(tmp_ann_test_error)
    opt_h_arr[i] = h_arr[index_h_opt]
    ann_test_error[i] = tmp_ann_test_error[index_h_opt]
    
    print("\n- ANN (h_opt="+str(opt_h_arr[i])+") : " + str(ann_test_error[i]))




# creating table 

df = pd.DataFrame({
    "Outer loop" : np.arange(1,k1+1),
    "opt_h" : opt_h_arr.astype(int),
    "ANN error" : ann_test_error,
    "opt_lambda" : opt_lambda_arr,
    "lmRegl error" : lmRegl_test_error,
    "lm error" : lm_test_error
})

print(df.head(n=k1))

# # plot table
# fig, ax = plt.subplots()
# ax.set_axis_off()
# fig.tight_layout()
# pd.plotting.table(ax,df, loc="center") 
# plt.show()

# df.to_csv("regression_B-Table.csv", index=False) # save table


# statistics test
regB_stat(ann_test_error,lmRegl_test_error, "ANN vs Reguralized Linear regression")
regB_stat(ann_test_error,lm_test_error, "ANN vs Baseline")
regB_stat(lmRegl_test_error,lm_test_error, "Reguralized Linear regression vs Baseline")



print("\n\n---Time : ",time.perf_counter()-t1)
    





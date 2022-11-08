import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Costum scripts
from load_data import load_data

from toolbox_02450 import rlr_validate

# Linear regression model with regularzation parameter to predict the Ozone level
K = 10

lambda_arr = sorted(np.append(np.power(10., range(-3,10)),np.array([30,55,150,250,400,600,1800,3000])))

data = load_data()
attrNames = list(data.columns)
data = np.array(data)
N,M = np.shape(data)
M -= 1 # minus the ozone attribute

# Split data into labels and attributes who will be used to predict the labels
y = data[:,0] # ozone levels (what we want to predict)
X = data[:,1:] # other attributes

# train the regularized linear regression
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambda_arr, K)


# Calc weights from optimal lambda
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 
w = np.linalg.solve(X.T@X+lambdaI , X.T@y)


# Printing results
print("Optimal lambda is : "+str(opt_lambda))
print("with error : ", opt_val_err)

print("\nWeights: ")
df = dict()
for i in range(M):
    df[attrNames[i+1]] = [w[i]]
df = pd.DataFrame.from_dict(df)
print(df.head())


# plotting
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambda_arr,mean_w_vs_lambda.T[:,1:],".-") # not plotting the bias term
plt.xlabel("Regularization factor")
plt.ylabel("Mean Coefficient Values")
plt.grid()

plt.subplot(1,2,2)
plt.loglog(lambda_arr,train_err_vs_lambda.T,"b.-",lambda_arr,test_err_vs_lambda.T,"r.-")
plt.xlabel("Regularization factor")
plt.ylabel("Squared error (crossvalidation)")
plt.legend(["Train error","Generalization(Valid.) error"])
plt.grid()

plt.show()


    





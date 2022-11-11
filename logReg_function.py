import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd

import sys

from load_data import load_data

data = np.array(load_data())

X = data[:, 1:]
y = data[:, 0]
N,M = np.shape(X)

y_label = []
for i in range(0, len(y)):
    if -1 < y[i] <= 6:
        y_label.append(0)
    elif 6 < y[i] <= 14:
        y_label.append(1)
    else:
        y_label.append(2)

y_label = np.array(y_label)


# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state=21)

opt_lambda = 10**(-8)

df = pd.DataFrame()

k = 0

w_sum_class0 = np.zeros(M)
w_sum_class1 = np.zeros(M)
w_sum_class2 = np.zeros(M)

for train_index, test_index in CV.split(X):
    k += 1
    print('Computing CV fold: {0}/{1}..'.format(k, K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y_label[train_index]
    X_test, y_test = X[test_index,:], y_label[test_index]

    mu = np.mean(X_train)
    sigma = np.std(X_train)
    X_train = (X_train-mu)/sigma
    X_test = (X_test-mu)/sigma   

    logis = LogisticRegression(penalty='l2', C=1 / opt_lambda, max_iter=4000, multi_class='multinomial')

    logis.fit(X_train, y_train)

    y_test_est = logis.predict(X_test).T

    test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

    w_sum_class0 += logis.coef_[0] # weights estimate for class 0 
    w_sum_class1 += logis.coef_[1] # weights estimate for class 1
    w_sum_class2 += logis.coef_[2] # weights estimate for class 2 

    # print(1/(1+np.exp(-(X_test[0,:] @ logis.coef_[0] ))))
    # print(1/(1+np.exp(-(X_test[0,:] @ logis.coef_[1] ))))
    # print(1/(1+np.exp(-(X_test[0,:] @ logis.coef_[2] ))))



print("\nAverage weights over K=10 CV : ")
print("class 0 : ", np.round(w_sum_class0/K,4))
print("class 1 : ", np.round(w_sum_class1/K,4))
print("class 2 : ", np.round(w_sum_class2/K,4))



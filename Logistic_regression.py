import numpy as np
from load_data import load_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree


font_size = 10
plt.rcParams.update({'font.size': font_size})

data = np.array(load_data())

X = data[:, 1:]
y = data[:, 0]

y_label = []

for i in range(0, len(y)):
    if -1 < y[i] <= 6:
        y_label.append(0)
    elif 6 < y[i] <= 14:
        y_label.append(1)
    else:
        y_label.append(2)

y_label = np.array(y_label)

K = 20

X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.1, stratify=y_label)

mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state=21)

k = 0
for train_index, test_index in CV.split(X):
    k += 1
    print('Computing CV fold: {0}/{1}..'.format(k, K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y_label[train_index]
    X_test, y_test = X[test_index,:], y_label[test_index]

    for g in range(0, len(lambda_interval)):
        logis = LogisticRegression(penalty='l2', C=1 / lambda_interval[g], max_iter=4000, multi_class='multinomial')

        logis.fit(X_train, y_train)

        # y_train_est = logis.predict(X_train).T
        y_test_est = logis.predict(X_test).T

        # train_error_rate[g] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[g] = np.sum(y_test_est != y_test) / len(y_test)

        # w_est = logis.coef_[0]
        # coefficient_norm[g] = np.sqrt(np.sum(w_est ** 2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    print(opt_lambda, min_error)


plt.figure(figsize=(10, 20))
plt.plot(np.log10(lambda_interval), train_error_rate*100)
plt.plot(np.log10(lambda_interval), test_error_rate*100)
# plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate * 100)
plt.semilogx(lambda_interval, test_error_rate * 100)
# plt.semilogx(opt_lambda, min_error * 100, 'o')
# plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error * 100, 2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda), 2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
# plt.legend(['Training error', 'Test error', 'Test minimum'], loc='upper right')

plt.show()

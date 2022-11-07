import numpy as np
import random
import sklearn.linear_model as lm
import sys
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import colorsys

# Costum scripts
from load_data import load_data

random.seed(42)

# Parameters
k1 = 10
k2 = 10

data = load_data()
N = data.shape[0]
data = np.array(data)

# Divide them into n_classes many evenly distributed ranges for Ozone levels and run PCA on the dataset
n_classes = 3
HSV_tuples = [(x*1.0/n_classes, 0.5, 0.5) for x in range(n_classes)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

arr = data[:,0]

# create histogram
d = dict()
for val in arr:
    val = str(val)
    if val in d:
        d[val] += 1
    else:
        d[val] = 1

# Divide values into n_classes many labels
i_arr = ["-1",]
summi = 0
sum_arr = []
limit = N/n_classes
keys_sorted = sorted(d.keys(), key=lambda x : int(x))
for k in keys_sorted:
    plt.bar(k, d[k]) # plot histogram of all histogram

    summi += d[k]
    if summi > limit or k==keys_sorted[-1]:
        i_arr.append(k)
        sum_arr.append(summi)
        summi=0

# print(i_arr) # print classes bins

# Plot histogram when divided into n_classes
plt.figure()
for i in range(1,n_classes+1):
    plt.bar(str(int(i_arr[i-1])+1)+"-"+i_arr[i],sum_arr[i-1],color=RGB_tuples[i-1] )
plt.title(str(n_classes)+" classes")
     
# Create label/coloring mask
mask_arr = np.zeros(N, dtype=int)
for j in range(N):
    for label,i_val in enumerate(i_arr[1:]):
        if arr[j]<=int(i_val):
            mask_arr[j]=label
            break

pca = PCA(n_components = 2)

data =  (data-np.mean(data,axis=0))/np.std(data,axis=0)# standardize the data 


# X = data[:,1:]
# y = data[:,0]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0., random_state = 0, shuffle=True)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

pca.fit(data)
data_trans = pca.transform(data)  # data points transformed on the first n_components=2 

# print(pca.explained_variance_ratio_)


# plot transformed data with their n_classes labels / ranges
plt.figure()
for l in range(n_classes):
    plt.scatter(data_trans[l==mask_arr,0],data_trans[l==mask_arr,1], color=RGB_tuples[l], label=i_arr[l]+"<x<="+i_arr[l+1])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
        





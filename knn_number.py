from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sys

filename = 'la_ozone.csv'
df = pd.read_csv(filename)

raw_data = df.values  
cols = range(1, 10) 
X = raw_data[:, cols]
N, M = X.shape

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,-1] # -1 takes the last column

y = raw_data[:,0]

for i in range(N):
    if y[i] <= 6:
        y[i] = 0
    elif 6 < y[i] <=14:
        y[i] = 1
    else:
        y[i] = 2
        
#KNeigbhorsClassifier from exercise 6.3.2
        
# Maximum number of neighbors
L=40

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state=(21))

errors = np.zeros((N,L))
i=0

df = pd.DataFrame()
pop_label = np.max(y)

for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    n_test = len(y_test)
    
    

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est!=y_test)
    
    #K index
    opt_index = np.argmin(errors[i,:])
    opt_kn = opt_index+1
    min_error = errors[i,opt_index]/n_test
    base_error = sum(y_test ==pop_label)/n_test

    i+=1

    df = df.append({"K":opt_kn, "Error":min_error, "baseline":base_error},ignore_index=True)

print(df.head(n=K))
    
    
print((100*sum(errors,0)/N))
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N, '-o')
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
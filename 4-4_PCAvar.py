import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd, solve

df = pd.read_csv("la_ozone.csv")
attr_names = {
    "ozone" : "Upland Maximum Ozone",
    "vh" : "Vandenberg 500 mb Height", 
    "wind" : "Wind Speed (mph)",
    "humidity" : "Humidity (%)",
    "temp" : "Sandburg AFB Temperature",
    "ibh" : "Inversion Base Height",
    "dpg" : "Daggot Pressure Gradient",
    "ibt" : "Inversion Base Temperature",
    "vis" : "Visibility (miles)",
    "doy" : "Day of the Year"  
}


df = df.drop(df.index[df["wind"]==21][0])  # delete the windspeed=21mph outlier row

# Standardize data
for attr in attr_names.keys():
    df[attr] = (df[attr]- np.mean(df[attr]))/np.std(df[attr])

Y = np.array(df) 

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
rho_cumulative = np.cumsum(rho)

threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),rho_cumulative,'o-')
plt.plot([0, len(rho)], [threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()

fig,ax = plt.subplots()
ax.table(cellText=np.round([rho, rho_cumulative], 4), loc="center", rowLabels=["Individual", "Cumulative"], colLabels=["PC "+str(i+1) for i in range(len(rho))] )
ax.axis('off')
ax.axis('tight')


# plot table with the first n_components eigen vectors of the attributes who explain the thershold=0.9 variance
n_components = np.where(rho_cumulative>threshold)[0][0]+1
fig,ax = plt.subplots()
ax.table(cellText=np.round(V[:,:n_components], 5), loc="center", rowLabels=list(attr_names.keys()), colLabels=["Comp. "+str(i+1) for i in range(n_components)] )
ax.axis('off')
ax.axis('tight')
 




# --------------- PCA 1 vs PCA 2 on ozone data -----------

V = V.T
Z = Y @ V

plt.figure()
for u in np.unique(Y): 
    mask = Y[:,0] == u        # plot every unique value in different color
    plt.scatter(Z[mask, 0], Z[mask, 1])

plt.title("Projected attributes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



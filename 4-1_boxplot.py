import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("la_ozone.csv")
attr_names = np.array(df.columns)

# Standardize the data i.e. mean=0 & std=1
for attr in attr_names:
    df[attr] = (df[attr] - np.mean(df[attr]))/np.std(df[attr])  
    
plt.boxplot(df, labels=attr_names, sym="r+", boxprops={"color" : "b"})
plt.title("Boxplot of standardized attributes")



plt.show()


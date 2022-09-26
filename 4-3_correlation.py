import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats


def round_sig(x,sig):
    x = str(round(x, sig-int(np.floor(np.log10(abs(x))))-1))
    if "e" in x: 
        return float(x[:np.min([sig+1,x.index("e")])] + x[x.index("e"):])  # sometimes it shows way more than significant digits
    else:
        return float(x)


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

attr_keys = list(attr_names.keys())

# source of limit choosing : https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1553-2712.1997.tb03646.x#:~:text=The%20reliability%20coefficient%20is%20the,least%200.80%20are%20considered%20desirable.
p_limit = 0.05
corrCoeff_limit = 0.8
# corr_table = []
for i in range(len(attr_keys)):
    for j in range(i+1,len(attr_keys)):
        # Linear correlation - Pearson
        corrCoeff_pearson, p = scipy.stats.mstats.pearsonr(df[attr_keys[i]], df[attr_keys[j]])
        p_pearson = round_sig(p, 5)
        corrCoeff_pearson = round_sig(corrCoeff_pearson, 5)

        # rank-order correlation - Spearman
        corrCoeff_spearman, p_spearman = scipy.stats.spearmanr(df[attr_keys[i]], df[attr_keys[j]])
        p_spearman = round_sig(p_spearman, 5)
        corrCoeff_spearman = round_sig(corrCoeff_spearman, 5)
    
        print("\n",attr_keys[i], " vs ", attr_keys[j]) 
        print("-- Pearon corrCoeff:" , corrCoeff_pearson, "- p-value:",p_pearson)
        print("-- Spearman corrCoeff:" , corrCoeff_spearman, "- p-value:",p_spearman)


        pearson_corr_bigger = corrCoeff_pearson >= corrCoeff_spearman # Linear correlation is greater

        if p_pearson <= p_limit and abs(corrCoeff_pearson) >= corrCoeff_limit and pearson_corr_bigger:
            print("-- LINEARLY CORRELATED")
            plt.figure()
            plt.scatter(df[attr_keys[i]], df[attr_keys[j]], color="b")
            plt.xlabel(attr_keys[i])
            plt.ylabel(attr_keys[j])
            plt.title("Linearly correlated - corrCoeff: " + str(corrCoeff_pearson) + " - p-value: "+ str(p_pearson))
        if p_spearman <= p_limit and abs(corrCoeff_spearman) >= corrCoeff_limit and not pearson_corr_bigger:
            print("-- RANK CORRELATED")
            plt.figure()
            plt.scatter(df[attr_keys[i]], df[attr_keys[j]], color="b")
            plt.xlabel(attr_keys[i])
            plt.ylabel(attr_keys[j])
            plt.title("Rank correlated - corrCoeff: " + str(corrCoeff_spearman) + " - p-value: "+ str(p_spearman))


# create table
# plt.figure("Result table")
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.axis('tight')
# ax.table(cellText=corr_table, loc="center", colLabels=["Attribute A", "Attribute B", "Correlation coefficient", "p-value", "Linearly correlated"])
plt.show()



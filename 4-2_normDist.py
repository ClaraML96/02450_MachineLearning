import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def round_sig(x,sig):
    x = str(round(x, sig-int(np.floor(np.log10(abs(x))))-1))
    if "e" in x: 
        return float(x[:sig+1] + x[x.index("e"):])  # sometimes it shows way more than significant digits
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
    "vis" : "Visibility (miles)"
    # "doy" : "Day of the Year"  # is a incrimental index
}

for attr, units in attr_names.items():

    ads, _ , _=scipy.stats.anderson(df[attr], dist="norm")
      
    # Calculate p-value for adjusted Anderson-Darling statistic 
    # source : R.B. D'Augostino and M.A. Stephens, Eds., 1986, Goodness-of-Fit Techniques, Marcel Dekker
    ads = ads*(1 + (.75/50) + 2.25/(50**2)) # adjusted
    print("\n"+attr)
    print("Adjusted Anderson-Darling statistic* = ", round(ads,4))
    if ads >= .6:
        p = np.exp(1.2937 - 5.709*ads - .0186*(ads*ads))
    elif ads >=.34:
        p = np.exp(.9177 - 4.279*ads - 1.38*(ads*ads))
    elif ads >.2:
        p = 1 - np.exp(-8.318 + 42.796*ads - 59.938*(ads*ads))
    else:
        p = 1 - np.exp(-13.436 + 101.14*ads - 223.73*(ads*ads))

    p = round_sig(p, 5)
    print("p = " , p)

    plt.figure(attr)
    plt.hist(df[attr], bins=len(np.unique(df[attr])), density=True, edgecolor='k', color="b")
    xmin,xmax = plt.xlim()
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, scipy.stats.norm.pdf(xx, np.mean(df[attr]), np.std(df[attr])), 'orange', linewidth=2)
    plt.xlabel(units)
    plt.ylabel("Frequency")
    plt.legend(["Normal fit", "Data"])

    if p >= 0.05:
        print("We ACCEPT the null hypothesis that the data is normally distributed")
        plt.title(attr + " - normally distributed - p-value=" + str(p))
    else:
        print("We REFUSE the null hypothesis, the data is not normally distributed")
        plt.title(attr + " - NOT normally distributed - p-value=" + str(p))

    # plt.savefig("p1_myndir/"+"4-2"+attr+"_freq.jpg")

plt.show()

 
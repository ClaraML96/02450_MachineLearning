import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv("la_ozone.csv")
    
    # Find and delete outlier
    del_arr = np.where(data["wind"]==21)[0]
    data = data.drop(labels=del_arr, axis=0)

    return data




if __name__ == "__main__":
    a = load_data()





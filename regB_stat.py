import pandas as pd
from scipy.stats import ttest_rel, t
import numpy as np


def regB_stat(err_arr1,err_arr2, title):
    # Compares two regression methods
    
    # 2-sided t-test 
    # - This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
    # - If we observe a large p-value, for example greater than 0.05 or 0.1 then we cannot reject the null hypothesis of identical average scores 
    #   if it is less then we reject the null hypothesis
    t_score, p_value = ttest_rel(err_arr1, err_arr2, alternative="two-sided")

    # Calculate the 95% confidence interval
    z_arr = err_arr1-err_arr2
    n = len(z_arr)
    z_tilde = np.mean(z_arr)
    sigma_tilde = np.sqrt(sum((z_arr-z_tilde)*(z_arr-z_tilde)/(n*(n-1))))
    cf95_L, cf95_U = t.interval(0.95, n-1, loc=z_tilde, scale=sigma_tilde)

    
    print("\n"+title)
    print("t-score : ", round(t_score,4))
    print("p-value : ", round(p_value,4))
    print("95% confidicene interval : [", round(cf95_L,4), " , ", round(cf95_U,4),"]" )


    if p_value <= 0.05:
        print("Null hypothesis ACCEPTED - the methods give the same results")
    else:
        print("Null hypothesis REJECTED - the methods DO NOT give the same results")

        if t_score < 0:
            print(title.split(" vs ")[1] + " has a larger error")
        else:
            print(title.split(" vs ")[0] + " has a larger error")

    output = {
        "Comparing" :  title,
        "t_score" : t_score,
        "p_value" : p_value,
        "cf95_L" : cf95_L,
        "cf95_U" : cf95_U
    }

    return output



if __name__ == "__main__":
    df = pd.read_csv("regression_B-Table.csv")

    df_table = pd.DataFrame()

    ann_error = df["ANN error"]
    rlr_error = df["lmRegl error"]
    lm_error = df["lm error"]


    df_table = df_table.append(regB_stat(ann_error,rlr_error, "ANN vs RLR"), ignore_index=True)
    df_table = df_table.append(regB_stat(ann_error,lm_error, "ANN vs LR"), ignore_index=True)
    df_table = df_table.append(regB_stat(rlr_error,lm_error, "RLR vs LM"), ignore_index=True)

    print("\n",df_table.head())



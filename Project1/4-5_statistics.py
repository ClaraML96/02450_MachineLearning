#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#Create a Dictionary of series
d = pd.read_csv("la_ozone.csv")
#Create a DataFrame
df = pd.DataFrame(d)

print(df.describe().round(2))

# 50% quantiles = median. 
#print('median:', df.median())

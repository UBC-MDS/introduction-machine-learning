import sys
sys.path.insert(0, 'exercises/')

import os
cwd = os.getcwd()
print(cwd)
print(os.listdir(cwd))
new = os.chdir("exercises/")
print(os.listdir(new))

from gini_impurity.py import gini2

import numpy as np
import pandas as pd

#from gini_impurity import gini2

# Loading in the data
candybar_df = pd.read_csv('data/candybars.csv', header=0, index_col=0)

candybar_df_binary = candybar_df[candybar_df['available_canada_america']!= "Both"]
print(candybar_df_binary.head())
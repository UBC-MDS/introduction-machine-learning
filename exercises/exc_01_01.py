import numpy as np
import pandas as pd


dataframe = pd.read_csv('data/candybars.csv', header=0, index_col=0)

print(candy_df.head())

candybar_feat = list(candy_df.columns)
print(candybar_feat)

candybar_names = list(candy_df.index)
print(candybar_names)

candybar_dim = candy_df.shape
print(candybar_dim)
import pandas as pd

fertile_df = pd.read_csv('data/fertility.csv')
print(fertile_df.iloc[:5, :5])
print(fertile_df.iloc[:5, 5:])


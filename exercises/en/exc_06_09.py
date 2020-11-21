import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Split the dataset
df_train, df_test = train_test_split(bball_df, test_size=0.2, random_state=7)

X_column = df_train[['country']]

# Build the tranformer and fit on it 
____ = ____(____, ____)
____.____(____);

# Transform the column country
____ = ____.____(____)

# Print the output of country_encoded
print(country_encoded)

# Let's look at this in a dataframe
pd.DataFrame(data=country_encoded,
                  columns=one_hot_encoder.get_feature_names(['country']),
                  index=X_column.index).head()
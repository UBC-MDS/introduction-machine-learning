import numpy as np
import pandas as pd
pd.options.display.max_columns = 5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Split the dataset
df_train, df_test = train_test_split(bball_df, test_size=0.2, random_state=7)

X_column = df_train[['country']]

# Build the tranformer and fit on it 
one_hot_encoder = OneHotEncoder(sparse=False, dtype='int')
one_hot_encoder.fit(X_column);

# Transform the column country
country_encoded = one_hot_encoder.transform(X_column)

# Print the output of country_encoded
print(country_encoded)

# Let's look at this in a dataframe
pd.DataFrame(data=country_encoded,
                  columns=one_hot_encoder.get_feature_names(['country']),
                  index=X_column.index).head()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ____
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

# Loading in the data
bball_df = pd.read_csv('data/bball_imp.csv').dropna(subset=['salary'])

# Split the dataset
df_train, df_test = train_test_split(bball_df, test_size=0.2, random_state=7)

X_train = df_train[["weight", "height", "draft_year", "draft_round",
                     "draft_peak", "team", "position", "country"]]
X_test = df_test[["weight", "height", "draft_year", "draft_round",
                     "draft_peak", "team", "position", "country"]]
y_train = df_train['salary']
y_test = df_test['salary']


# Split the numeric and categorical features 
numeric_features = [ "weight",
                     "height",
                     "draft_year",
                     "draft_round",
                     "draft_peak"]

categorical_features = ["team", "position", "country"]

# Build a numeric pipeline
____ = ____(
    steps=[(____, ____), 
           (____, ____)]
)

# Build a categorical pipeline
____ = ____(
    steps=[(____, ____),
           (, ____)]
)

# Build a column transformer
____ = ____(
    transformers=[
        (____, ____, ____),
        (____, ____, ____)
    ] 
)

# Build a main pipeline
____ = ____(
    steps=[
        (____, ____),
        (____, ____)])

# Cross validate
scores = cross_validate(____, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
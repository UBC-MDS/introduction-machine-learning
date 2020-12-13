import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

# Build a categorical pipeline
categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Build a column transformer
col_transformer = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ] 
)

# Build a main pipeline
main_pipe = Pipeline(
    steps=[
        ("preprocessor", col_transformer),
        ("reg", KNeighborsRegressor())])

# Cross validate
scores = cross_validate(main_pipe, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
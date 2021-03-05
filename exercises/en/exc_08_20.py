import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

# Loading in the data
bball = pd.read_csv('data/bball.csv')
bball = bball[(bball['draft_year'] != 'Undrafted') & (bball['draft_round'] != 'Undrafted') & (bball['draft_peak'] != 'Undrafted')]
bball = bball.replace({'F-G': 'Other', 'F-C': 'Other', 'G-F': 'Other', 'C-F': 'Other', 'C': 'Other'})
df_train, df_test = train_test_split(bball, test_size=0.2, random_state=1)


X_train = df_train[["weight", "height", "draft_year", "draft_round",
                     "draft_peak", "team", "salary", "country"]]
X_test = df_test[["weight", "height", "draft_year", "draft_round",
                     "draft_peak", "team", "salary", "country"]]
y_train = df_train['position']
y_test = df_test['position']


# Split the numeric and categorical features 
numeric_features = [ "weight",
                     "height",
                     "draft_year",
                     "draft_round",
                     "draft_peak"]

categorical_features = ["team", "country"]


# Build a numeric pipeline
numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"))

# Build a categorical pipeline
categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))


# Build a numeric pipeline
numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

# Build a categorical pipeline
categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

# Build a categorical transformer
col_transformer = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features))

# Build a main pipeline
____ = ____

# Fit your pipeline on the training set
____.____

# Plot your confusion matrix on your test set 
____;
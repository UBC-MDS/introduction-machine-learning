import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

# Loading in the data
bball = pd.read_csv('data/bball_cm.csv')

train_df, test_df = train_test_split(bball, test_size=0.2, random_state=1)

X_train = train_df.drop(columns=['full_name', 'jersey', 'b_day', 'college', 'position'])
y_train = train_df['position']
X_test = test_df.drop(columns=['full_name', 'jersey', 'b_day', 'college', 'position'])
y_test = test_df['position']

drop_features = ['full_name', 'jersey', 'b_day', 'college', 'position']

numeric_features = [
    "rating",
    "height",
    "weight",
    "salary",
    "draft_year",
    "draft_round",
    "draft_peak"]

categorical_features = [
    "team",
    "country"]

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    ("drop", drop_features),
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features)
)

# Build a pipeline containing the column transformer and an SVC model
pipe_bb = make_pipeline(preprocessor, SVC())

# Fit your pipeline on the training data
pipe_bb.fit(X_train, y_train);

# Plot your confusion matrix and "show it"
plot_confusion_matrix(pipe_bb, X_test, y_test,
                      cmap="PuRd");
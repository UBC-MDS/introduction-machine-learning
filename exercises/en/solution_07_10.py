import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier,
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Loading in the data
bball = pd.read_csv('data/bball.csv')
bball = bball[(bball['draft_year'] != 'Undrafted') & (bball['draft_round'] != 'Undrafted') & (bball['draft_peak'] != 'Undrafted')]

train_df, test_df = train_test_split(bball, test_size=0.2, random_state=1)

X_train_big = train_df.drop(columns=['full_name', 'jersey',
                                     'b_day', 'college', 'position'])
y_train_big = train_df['position']
X_test = test_df.drop(columns=['full_name', 'jersey',
                               'b_day', 'college', 'position'])
y_test = test_df['position']

X_train, X_valid, y_train, y_valid = train_test_split(X_train_big, 
                                                      y_train_big, 
                                                      test_size=0.3, 
                                                      random_state=123)
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
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features)
)

# Build a pipeline containing the column transformer and an SVC model
pipe_bb = make_pipeline(preprocessor, SVC())

# Fit your pipeline on the training data
pipe_bb.fit(X_train, y_train);

# Predict your values on the validation set
# Save them in an object named predicted_y
predicted_y = pipe_bb.predict(X_valid)

# Using sklearn tools, print a classification_report
print(classification_report(y_valid, predicted_y, digits=3, zero_division=0))




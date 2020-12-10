import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Loading in the data
bball = pd.read_csv('data/bball_cm.csv')

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

categorical_features = ["team", "country"]

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features))

# Build a pipeline containing the column transformer and an SVC model 
# Name it pipe_unbalanced and fit it on the training data
pipe_unbalanced = make_pipeline(preprocessor, SVC())
pipe_unbalanced.fit(X_train, y_train);

# Predict your values on the validation set
# Save them in an object named predicted_y
predicted_y = pipe_unbalanced.predict(X_valid)

# Using sklearn tools, calculate precision
# Save it in an object named precision
precision = precision_score(y_valid, predicted_y, pos_label="F").round(3)
print("precision: ", precision)

# Using sklearn tools, calculate recall
# Save it in an object named recall
recall = recall_score(y_valid, predicted_y, pos_label="F").round(3)
print("recall: ", recall)

# Using sklearn tools, calculate f1
# Save it in an object named f1
f1 = f1_score(y_valid, predicted_y, pos_label="F").round(3)
print("f1:", f1)

# Using sklearn tools, print a classification_report
print(classification_report(y_valid, predicted_y, digits=3))




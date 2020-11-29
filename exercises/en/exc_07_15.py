import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Loading in the data
pk_df = pd.read_csv('data/pokemon.csv')

train_df, test_df = train_test_split(pk_df, test_size=0.2, random_state=1)

X_train_big = train_df.drop(columns=['legendary'])
y_train_big = train_df['legendary']
X_test = test_df.drop(columns=['legendary'])
y_test = test_df['legendary']

X_train, X_valid, y_train, y_valid = train_test_split(X_train_big, 
                                                      y_train_big, 
                                                      test_size=0.3, 
                                                      random_state=123)

numeric_features = ["deck_no",  
                    "attack",
                    "defense" ,
                    "sp_attack",
                    "sp_defense",
                    "speed",
                    "capture_rt",
                    "total_bs"]

categorical_features = [
    "type"]

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
# Name this pipeline pipe_unbalanced
____ = ____

# Fit your unbalanced pipeline on the training data
____.____

# Predict your values on the validation set
# Save them in an object named unbalanced_predicted
____ = ____

# Using sklearn tools, print a classification_report from the validation set
print(____(____))

# Build another pipeline containing the column transformer and an SVC model
# This time use the parameter class_weight="balanced"
# Name this pipeline pipe_balanced
____ = ____

# Fit your balanced pipeline on the training data
____.____

# Predict your values on the validation set
# Save them in an object named balanced_predicted
____ = ____

# Using sklearn tools, print a classification_report from the validation set
print(____(____))
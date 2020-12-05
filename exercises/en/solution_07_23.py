import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

# Loading in the data
pk_df = pd.read_csv('data/pokemon.csv')

train_df, test_df = train_test_split(pk_df, test_size=0.2, random_state=1)

X_train = train_df.drop(columns=['legendary'])
y_train = train_df['legendary']
X_test = test_df.drop(columns=['legendary'])
y_test = test_df['legendary']

numeric_features = ["deck_no",  
                    "attack",
                    "defense" ,
                    "sp_attack",
                    "sp_defense",
                    "speed",
                    "capture_rt",
                    "total_bs"]

categorical_features = ["type"]

numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features))

# Build a pipeline containing the column transformer and an SVC model
# Use the parameter class_weight="balanced"
# Name this pipeline main_pipe
main_pipe = make_pipeline(preprocessor, SVC(class_weight="balanced"))

# Perform cross validation on the training split using the scoring measures accuracy, precision and recall 
# Save the results in a dataframe named multi_scores
multi_scores = pd.DataFrame(cross_validate(main_pipe,
                                           X_train,
                                           y_train,
                                           return_train_score=True,
                                           scoring = ['accuracy', 'precision', 'recall']))
multi_scores

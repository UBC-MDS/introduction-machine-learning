import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import ____

# Loading in the data
pk_df = pd.read_csv('data/pokemon.csv')

train_df, test_df = train_test_split(pk_df, test_size=0.2, random_state=1)

X_train = train_df.drop(columns=['legendary'])
y_train = train_df['legendary']
X_test = test_df.drop(columns=['legendary'])
y_test = test_df['legendary']


numeric_features = ["attack",
                    "defense" ,
                    "sp_attack",
                    "sp_defense",
                    "speed",
                    "capture_rt"]

drop_features = ["type", "deck_no", "gen", "name", "total_bs"]

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

preprocessor = make_column_transformer(
    ("drop", drop_features),
    (numeric_transformer, numeric_features))

# Build a pipeline containing the column transformer and a Logistic Regression model
# use the parameter class_weight="balanced"
# Name this pipeline pkm_pipe
____ = ____

# Fit your pipeline on the training data
____;

# Score your model on the test set 
# Save this in an object named lr_scores
____ = ____
print("logistic Regression Test Score:", lr_scores)

# Fill in the blanks below to asses the model's feature coefficients. 
pkm_coefs = pd.DataFrame({'features':____, 'coefficients':____['logisticregression'].____[0]})
pkm_coefs
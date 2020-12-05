import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier

# Loading in the data
fertile_df = pd.read_csv('data/fertility.csv').dropna(subset=['diagnosis'])

# Split the dataset
df_train, df_test = train_test_split(fertile_df, test_size=0.2, random_state=7)

X_train = df_train.drop(columns='diagnosis')
X_test = df_test.drop(columns='diagnosis')
y_train = df_train['diagnosis']
y_test = df_test['diagnosis']


# Split the numeric and categorical features 

# What are the numeric features? 
# Add them to a list named numeric_features
numeric_features = [ 'age', 'sitting_hrs']

# What are the binary features? 
# Add them to a list named binary_features
binary_features = ['childish_diseases', 'accident_trauma', 'surgical_intervention']

# What are the ordinal features? 
# Add them to a list named ordinal_features
ordinal_features = ['high_fevers_last_year', 'freq_alcohol_con', 'smoking_habit']

# What are the rest of the categorical features? 
# Add them to a list named categorical_features
categorical_features = ["season"]

# Order the values in high_fevers_last_year and name the list fever_order
# The options are 'more than 3 months ago', 'less than 3 months ago' and 'no'
fever_order = ['no', 'more than 3 months ago', 'less than 3 months ago']

# Order the values in smoking_habit and name the list smoking_order
# The options are 'occasional', 'daily' and 'never'
smoking_order = [ 'never', 'occasional', 'daily']

# Order the values in freq_alcohol_con and name the list alcohol_order
# The options are 'once a week', 'hardly ever or never', 'several times a week', 
# 'several times a day' and 'every day'
alcohol_order = ['hardly ever or never', 'once a week',
                 'several times a week', 'every day', 'several times a day']

# Pipelines

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

binary_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(drop="if_binary", dtype=int)
    )

ordinal_transformer1 = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[fever_order], dtype=int)
)

ordinal_transformer2 = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[smoking_order], dtype=int)
)

ordinal_transformer3 = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[alcohol_order], dtype=int)
)

# Column transformer
preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (ordinal_transformer1, ['high_fevers_last_year']),
  		(ordinal_transformer2, ['smoking_habit']),
 		(ordinal_transformer3, ['freq_alcohol_con']),
        (binary_transformer, binary_features)
)

# Build a main pipeline using KNeighborsClassifier and name it main_pipe
main_pipe = make_pipeline(preprocessor, KNeighborsClassifier())

# Cross validate
scores = cross_validate(main_pipe, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
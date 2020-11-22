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
____ = ____

# What are the binary features? 
# Add them to a list named binary_features
____ = ____

# What are the ordinal features? 
# Add them to a list named ordinal_features
____ = ____

# What are the rest of the categorical features? 
# Add them to a list named categorical_features
____ = ____

# Order the values in high_fevers_last_year and name the list fever_order
# The options are 'more than 3 months ago', 'less than 3 months ago' and 'no'
____ = ____

# Order the values in smoking_habit and name the list smoking_order
# The options are 'occasional', 'daily' and 'never'
____ = ____

# Order the values in freq_alcohol_con and name the list alcohol_order
# The options are 'once a week', 'hardly ever or never', 'several times a week', 
# 'several times a day' and 'every day'
____ = ____

# Pipelines
____ = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())


____ = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))


____ = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(drop="if_binary", dtype=int)
    )


____ = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[fever_order], dtype=int)
)


____ = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[smoking_order], dtype=int)
)

____ = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[alcohol_order], dtype=int)
)

# Column transformer
preprocessor = make_column_transformer(
        (____, numeric_features),
        (categorical_transformer, ____),
        (____, ['high_fevers_last_year']),
  		(____, ['smoking_habit']),
 		(____, ['freq_alcohol_con']),
        (binary_transformer, ____)
)

# Build a main pipeline using KNeighborsClassifier and name it main_pipe
____ = ____(____, ____)

# Cross validate
scores = cross_validate(____, ____, ____, return_train_score=True)
pd.DataFrame(scores)
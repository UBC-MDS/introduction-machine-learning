import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import ____
import scipy

# Loading in the data
tweets_df = pd.read_csv('data/balanced_tweets.csv').dropna(subset=['target'])

# Split the dataset into the feature table `X` and the target value `y`
____ = ____
____ = ____

# Split the dataset into X_train, X_test, y_train, y_test 
____ = ____


param_grid = {
    "countvectorizer__max_features": range(1,1000)
}

# Make a pipeline with CountVectorizer as the first step and KNeighborsClassifier as the second 
____ = ____

# perform RandomizedSearchCV using the parameters specified in param_grid
____ = ____(____, ____, n_jobs=-1, cv=5, return_train_score=True, n_iter=10)
____.____(____)

## What is the best max_features value? Save it in an object name tweet_feats
____ = ____
print(tweet_feats)

## What is the best score? Save it in an object named tweet_val_score
____ = ____
print(tweet_val_score)

# Score the optimal model on the test set and save it in an object named tweet_test_score
____ = ____
print(tweet_test_score)


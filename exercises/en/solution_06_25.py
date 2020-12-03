import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import scipy

# Loading in the data
tweets_df = pd.read_csv('data/balanced_tweets.csv').dropna(subset=['target'])

# Split the dataset into the feature table `X` and the target value `y`
X = tweets_df['text']
y = tweets_df['target']

# Split the dataset into X_train, X_test, y_train, y_test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)


param_grid = {
    "countvectorizer__max_features": range(1,1000)
}

# Make a pipeline with CountVectorizer as the first step and SVC as the second 
pipe = make_pipeline(CountVectorizer(), SVC())

# perform RandomizedSearchCV using the parameters specified in param_grid
tweet_search = RandomizedSearchCV(pipe, param_grid, n_jobs=-1, cv=5, return_train_score=True, n_iter=10)
tweet_search.fit(X_train, y_train)

## What is the best max_features value? Save it in an object name tweet_feats
tweet_feats = tweet_search.best_params_['countvectorizer__max_features']
print(tweet_feats)

## What is the best score? Save it in an object named tweet_val_score
tweet_val_score = tweet_search.best_score_
print(tweet_val_score)

# Score the optimal model on the test set and save it in an object named tweet_test_score
tweet_test_score = tweet_search.score(X_test, y_test)
print(tweet_test_score)


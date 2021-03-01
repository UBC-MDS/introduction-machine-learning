import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
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

categorical_features = ["type"]

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), 
    StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features))
    
param_grid = {"logisticregression__C": scipy.stats.uniform(0, 100)}

# Build a pipeline containing the column transformer and a Logistic Regression model
# use the parameter class_weight="balanced" and set max_iter=1000
# Name this pipeline pkm_pipe
pkm_pipe = make_pipeline(preprocessor, LogisticRegression(class_weight="balanced", max_iter=1000))

# Perform RandomizedSearchCV using the parameters specified in param_grid
# Use n_iter equal to 10, 5 cross-validation folds and return the training score. 
# Name this object pmk_search
pmk_search = RandomizedSearchCV(pkm_pipe, param_grid,
                                n_jobs=-1, cv=5, return_train_score=True,
                                 n_iter=10, scoring = 'f1', random_state=2028)

# Train your pmk_search on the training data
pmk_search.fit(X_train, y_train)

# What is the best C value? Save it in an object name pkm_best_c
pkm_best_c= pmk_search.best_params_['logisticregression__C']
print("Best C value:", pkm_best_c)

# What is the best f1 score? Save it in an object named pkm_best_score
pkm_best_score = pmk_search.best_score_
print("Best f1 score:", pkm_best_score)

# Find the predictions of the test set using predict. 
# Save this in an object named predicted_y
predicted_y = pmk_search.predict(X_test)

# Find the target class probabilities of the test set using predict_proba. 
# Save this in an object named proba_y
proba_y = pmk_search.predict_proba(X_test)

# This next part has been done for you
lr_probs = pd.DataFrame({
             "Pokemon": test_df['name'],
             "true y":y_test, 
             "pred y": predicted_y.tolist(),
             "prob_legend": proba_y[:, 1].tolist()})
             
# Take the dataframe lr_probs and sort them in descending order of the models confidence
# in predicting legendary pokemon
# Save this in an object named legend_sorted       
legend_sorted = lr_probs.sort_values(by='prob_legend', ascending=False)
legend_sorted
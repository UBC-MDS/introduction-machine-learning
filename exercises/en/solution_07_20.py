import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error, r2_score


# Loading in the data
bball = pd.read_csv('data/bball.csv')
bball = bball[(bball['draft_year'] != 'Undrafted') & (bball['draft_round'] != 'Undrafted') & (bball['draft_peak'] != 'Undrafted')]

train_df, test_df = train_test_split(bball, test_size=0.2, random_state=1)

X_train_big = train_df.drop(columns=['full_name', 'jersey', 'b_day', 'college','salary'])
y_train_big = train_df['salary']
X_test = test_df.drop(columns=['full_name', 'jersey', 'b_day', 'college', 'salary'])
y_test = test_df['salary']

X_train, X_valid, y_train, y_valid = train_test_split(X_train_big, 
                                                      y_train_big, 
                                                      test_size=0.3, 
                                                      random_state=123)
numeric_features = [
    "height",
    "weight",
    "draft_year",
    "draft_round",
    "draft_peak"]

categorical_features = [
    "team",
    "country",
    "position"]

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
pipe_bb = make_pipeline(preprocessor, SVR())

# Fit your pipeline on the training data
pipe_bb.fit(X_train, y_train);

# Using your model, find the predicted values of the validation set
# Save them in an object named predict_valid
predict_valid = pipe_bb.predict(X_valid)

# Calculate the MSE and save the result in an object named mse_calc
mse_calc = mean_squared_error(y_valid, predict_valid)
print(mse_calc)

# Calculate the RMSE and save the result in an object named rmse_calc
rmse_calc = np.sqrt(mean_squared_error(y_valid, predict_valid))
print(rmse_calc)

# Calculate the R^2 and save the result in an object named r2_calc
r2_calc = r2_score(y_valid, predict_valid)
print(r2_calc)

# Calculate the MAPE and save the result in an object named mape_calc
mape_calc = np.mean(np.abs((predict_valid - y_valid) / y_valid)) * 100.0
print(mape_calc)
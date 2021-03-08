import numpy as np
import pandas as pd
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import  mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.metrics import make_scorer
from sklearn.linear_model import ____

# Loading in the data
bball = pd.read_csv('data/bball.csv')
bball = bball[(bball['draft_year'] != 'Undrafted') & (bball['draft_round'] != 'Undrafted') & (bball['draft_peak'] != 'Undrafted')]

train_df, test_df = train_test_split(bball, test_size=0.2, random_state=1)

X_train = train_df[['height']]
y_train = train_df['weight']
X_test = test_df[['height']]
y_test = test_df['weight']

## Define mape function 
def mape(true, pred):
    return 100.*np.mean(np.abs((pred - true) / true))

## Create a mape scorer where lower number are better 
neg_mape_scorer = ____(____, ____=____)

# Create a set of values for alpha
param_dist = {
    "alpha": [0.1, 1, 10, 100, 1000, 10000]}

# Build a Ridge model called ridge_bb
____ = ____

## Use GridSearchCV to hyperparameter tune. 
grid_search = GridSearchCV(
    ____, ____, cv=5,
     n_jobs=-1,
    ____=____)

# Fit your grid search on the training data
____.____(X_train, y_train)

# What is the best value for alpha?
# Save it in an object named best_alpha
____ = ____
print(best_alpha)

# What is the best MAPE score?
# Save it in an object named best_mape
____ = ____
print(best_mape)

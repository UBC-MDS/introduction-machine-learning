import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Ridge
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.metrics import make_scorer

# Loading in the data
bball = pd.read_csv('data/bball.csv')
bball = bball[(bball['draft_year'] != 'Undrafted') & (bball['draft_round'] != 'Undrafted') & (bball['draft_peak'] != 'Undrafted')]

train_df, test_df = train_test_split(bball, test_size=0.2, random_state=1)

X_train = train_df[['height', 'weight']]
y_train = train_df['salary']
X_test = test_df[['height', 'weight']]
y_test = test_df['salary']

player_stats = pd.DataFrame([[2.05,93.2]], columns =['height', 'weight'])

# Build a Ridge model called ridge_bb
____ = ____

# Fit your grid search on the training data
____

# What are the coefficients for this model?
# Save these in an object named bb_weights

____ = ____
print(bb_weights)

# What is the intercept for this model? 
# Save this in an object named bb_intercept

____ = ____
print(bb_intercept)

# Using the weights and intercept discovered above, 
# calculate the models's prediction
# Save it in an object named player_predict

____ = ____
print(player_predict)

# Check your answer using predict
____
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ____

# Loading in the data
pokemon = pd.read_csv('data/pokemon.csv')

# Define X and y
X = pokemon.loc[:, 'attack':'capture_rt']
y = pokemon['legendary']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33)

# Create a model
____ = ____

# Cross validate
____ = ____

# Covert scores into a dataframe
____ = ____

# Calculate the mean value of each column
____ = ____

# Display each score mean value
____
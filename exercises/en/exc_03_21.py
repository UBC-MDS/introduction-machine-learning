import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Loading in the data
pokemon = pd.read_csv('data/pokemon.csv')

# Define X and y
X = pokemon.loc[:, 'speed':'capture_rt']
y = pokemon['legendary']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Create a model
____ = ____

# Fit your data 
____.____

# Score the model on the test set 
test_score = ____.____

print("The test error: " + str(test_error))


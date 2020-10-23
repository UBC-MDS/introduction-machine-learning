import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Create a model
model = DecisionTreeClassifier(max_depth=4)

# Fit your data 
model.fit(X_train,y_train)

# Score the model on the test set 
test_error = round(model.score(X_test, y_test), 4)

print("The test error: " + str(test_error))


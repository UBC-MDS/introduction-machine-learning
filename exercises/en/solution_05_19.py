import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Loading in the data
bball_df = pd.read_csv('data/bball.csv')
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]

# Define X and y
X = bball_df.loc[:, ['height', 'weight', 'salary']]
y = bball_df['position']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Build a pipeline named bb_pipe
bb_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler()),
                   ("knn", KNeighborsClassifier()),
                ])

# Build a grid of the parameters you wish to search. 
param_grid = {
    "knn__n_neighbors" : [1, 5, 10, 20, 30, 40, 50],
    "knn__weights" : ['uniform', 'distance']
}


grid_search = GridSearchCV(bb_pipe, param_grid, verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)


best_hyperparams = grid_search.best_params_


best_model_pipe = random_search.best_estimator_
best_model_pipe.fit(X_train, y_train)

best_model_pipe.score(X_test, y_test)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate

# Loading in the data
hockey_df = pd.read_csv('data/canucks.csv')
hockey_df =hockey_df[hockey_df['Position'] != 'Goalie']

# Define X and y
X = hockey_df.loc[:, ['Age', 'Height', 'Weight', 'Experience', 'Salary']]
y = hockey_df['Position']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7)

# Create a model
model = DecisionTreeClassifier(max_depth=2)


# Cross Validate and fit 
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)

# Covert scores into a dataframe
scores_df = pd.DataFrame(scores)

# Calculate the mean value of each column
mean_scores = scores_df.mean()

# Display each score mean value 
# Remember that in this case "test_score" is actually "validation" score

mean_scores

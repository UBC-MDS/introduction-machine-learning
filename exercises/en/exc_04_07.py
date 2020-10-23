import pandas as pd
from math import ____

# Loading in the data
pokemon_df = pd.read_csv('data/pokemon.csv')

# Define X and y
X = pokemon_df.drop(columns = ['deck_no', 'name','total_bs', 'type', 'legendary'])
y = pokemon_df['legendary']


# Subtract the two first pokemon feature vectors and save it in an object name sub_pk
____ = ____

# Square the difference and save it in an object named sq_sub_pk
____ = ____

# Sum the squared difference from each dimension and save the result in an object named sss_pk
____ = ____

# Finally, take the square root of the entire calculation and save it in an object named pk_distance
____ = ____

____
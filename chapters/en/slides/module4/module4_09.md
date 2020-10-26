---
type: slides
---

# Finding the nearest neighbour

Notes: <br>

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
train_df.head(3)
```

```out
     longitude  latitude country
160   -76.4813   44.2307  Canada
127   -81.2496   42.9837  Canada
169   -66.0580   45.2788  Canada
```

``` python
dists = euclidean_distances(train_df[["latitude", "longitude"]])
dists
```

```out
array([[ 0.        ,  4.92866046, 10.47586257, ..., 45.36619339,  3.13968038,  9.58476504],
       [ 4.92866046,  0.        , 15.36399019, ..., 40.48484175,  1.80868018, 14.45684087],
       [10.47586257, 15.36399019,  0.        , ..., 55.83947468, 13.60621684,  0.94361393],
       ...,
       [45.36619339, 40.48484175, 55.83947468, ...,  0.        , 42.23325838, 54.93872568],
       [ 3.13968038,  1.80868018, 13.60621684, ..., 42.23325838,  0.        , 12.70774745],
       [ 9.58476504, 14.45684087,  0.94361393, ..., 54.93872568, 12.70774745,  0.        ]])
```

``` python
dists.shape
```

```out
(167, 167)
```

Notes:

Now that we know how to calculate the Euclidean distance between
examples, let see how close all the cities are to all other cities in
the dataset.

---

``` python
pd.DataFrame(dists).loc[:5,:5]
```

```out
           0          1          2          3          4          5
0   0.000000   4.928660  10.475863   3.402295   9.046000  44.329135
1   4.928660   0.000000  15.363990   8.326614  13.965788  39.839439
2  10.475863  15.363990   0.000000   7.195350   2.653738  54.549042
3   3.402295   8.326614   7.195350   0.000000   5.643921  47.391337
4   9.046000  13.965788   2.653738   5.643921   0.000000  52.532333
5  44.329135  39.839439  54.549042  47.391337  52.532333   0.000000
```

``` python
np.fill_diagonal(dists, np.inf)
pd.DataFrame(dists).loc[:5,:5]
```

```out
           0          1          2          3          4          5
0        inf   4.928660  10.475863   3.402295   9.046000  44.329135
1   4.928660        inf  15.363990   8.326614  13.965788  39.839439
2  10.475863  15.363990        inf   7.195350   2.653738  54.549042
3   3.402295   8.326614   7.195350        inf   5.643921  47.391337
4   9.046000  13.965788   2.653738   5.643921        inf  52.532333
5  44.329135  39.839439  54.549042  47.391337  52.532333        inf
```

Notes:

It’s important that we use the `fill_diagonal()` tool to replace all the
diagonal entries which is the distance from a city to itself.

It makes sense that there those entries have a distance of 0 but if we
are trying to find the city with the minimum distance, we would never be
getting the closest neighbour and instead be getting itself each time.

This is why we “fill diagonal” entries with a very large number,
infinity in fact.

---

Feature vector for city 0:

``` python
train_df.iloc[0]
```

```out
longitude   -76.4813
latitude     44.2307
country       Canada
Name: 160, dtype: object
```

Distances from city 0 to 5 other cities:

``` python
dists[0][:5]
```

```out
array([        inf,  4.92866046, 10.47586257,  3.40229467,  9.04600003])
```

Notes: Next, let’s look at the distances between City 0 and the first 5
cities in the dataframe.

---

``` python
train_df.iloc[[0]]
```

```out
     longitude  latitude country
160   -76.4813   44.2307  Canada
```

``` python
np.argmin(dists[0])
```

```out
157
```

``` python
train_df.iloc[[157]]
```

```out
    longitude  latitude country
96   -76.3019    44.211  Canada
```

``` python
dists[0][157]
```

```out
0.1804783920605758
```

Notes:

We can find the closest city to city 0 with
<a href="https://numpy.org/doc/stable/reference/generated/numpy.argmin.html" target="_blank">`np.argmin`</a>.

The closest city from city 0 is city 157 in the training dataframe.

Its feature vector tells us that it’s city 96 (of the whole dataset) and
is located at longitude -76.3019 and latitude 44.211 in country Canada.

That’s 0.18 units away from city 0.

---

### Finding the distances to a query point

``` python
query_point = [[-80, 25]]
```

``` python
dists = euclidean_distances(train_df[["longitude", "latitude"]], query_point)
dists[0:5]
```

```out
array([[19.54996348],
       [18.02706204],
       [24.60912622],
       [21.39718237],
       [25.24111312]])
```

We can find the city closest to the query point (-80, 25) using:

``` python
np.argmin(dists)
```

```out
147
```

The distance between the query point and closest city is:

``` python
dists[np.argmin(dists)].item()
```

```out
3.8383922936564634
```

Notes:

We can also find the distances to a new “test” or “query” city.

This time we add a second argument in `euclidean_distances()` which has
the coordinates of our query point.

This produces an array with the distances from each city to the query
point we specified.

---

``` python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(train_df[['longitude', 'latitude']]);
nn.kneighbors([[-80, 25]])
```

```out
(array([[3.83839229]]), array([[147]]))
```

Notes:

You can do the same thing using Sklearn’s NearestNeighbors function and
we get the same thing\!

All this matches our intuition of “distance” in the real world.

And we could also extend it to points in 3D space.

In fact, we can extend it to arrays (“vectors”) of any length.

---

``` python
pokemon_df = pd.read_csv("data/pokemon.csv")
X = pokemon_df.drop(columns = ['deck_no', 'name','total_bs', 'type', 'legendary'])
y = pokemon_df[['legendary']]
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train.head()
```

```out
     attack  defense  sp_attack  sp_defense  speed  capture_rt  gen
362      40       50         55          50     25         255    3
132      55       50         45          65     55          45    1
704      75       53         83         113     60          45    6
9        30       35         20          20     45         255    1
687      52       67         39          56     50         120    6
```

Note:

Let’s look at our pokemon data which will have 7 dimensions.

---

``` python
dists = euclidean_distances(X_train[:3])
dists
```

```out
array([[  0.        , 213.4338305 , 226.54138695],
       [213.4338305 ,   0.        ,  64.86139067],
       [226.54138695,  64.86139067,   0.        ]])
```

``` python
dists[0,2]
```

```out
226.54138694728607
```

``` python
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train);
nn.kneighbors(X_test.iloc[[1]])
```

```out
(array([[15.5241747]]), array([[143]]))
```

``` python
X_test.to_numpy().shape
```

```out
(161, 7)
```

Notes:

The distance between pokemon 0 and pokemon 2 can be found with
`dists[0,2]`.

We can find the most similar Pokemon from our training data to Pokemon 1
from the test set using `NearestNeighbors` from the Sklearn package.

---

``` python
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train);
nn.kneighbors(X_test.iloc[1])
```

``` out
ValueError: Expected 2D array, got 1D array instead:
array=[605  55  55  85  55  30 255 335   5].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```

``` python
X_test.iloc[1].shape
```

```out
(7,)
```

``` python
X_test.iloc[[1]].shape
```

```out
(1, 7)
```

``` python
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train);
nn.kneighbors(X_test.iloc[[1]])
```

```out
(array([[15.5241747 , 25.90366769, 27.91057147, 33.3166625 , 34.69870315]]), array([[143, 364, 515, 638,   0]]))
```

Notes:

We can also find the 5 most similar Pokemon in the training data to test
Pokemon 1.

We need to be careful though.

A numpy array with shape (9,) is 1 dimensional whereas (1, 9) is 2
dimensional which is what `kneighbors()` needs as an input.

Now we can see the top 5 most similar pokemon.

---

# Let’s apply what we learned\!

Notes: <br>

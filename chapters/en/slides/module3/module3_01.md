---
type: slides
---

# Data Splitting

Notes: <br>

---

## Recap

### Training error versus Generalization error

  - Given a model 𝑀, in Machine Learning (ML), people usually talk about
    two kinds of errors of 𝑀:

<!-- end list -->

1.  Error on the training data
    <img src="/module2/trainning_e.gif"  width = "12%" alt="404 image" />

2.  Error on the entire distribution 𝐷 of data
    <img src="/module2/d_e.gif"  width = "9%" alt="404 image" />

Notes:

At the end of module 2, we discussed two kinds of errors in a model.

1.  Error on the training data: \(error_{training}(M)\)
2.  Error on the entire distribution \(D\) of data: \(error_{D}(M)\)

But we do not have access to the entire distribution which is where our
interests lie so what do we do?

---

## We can approximate generalization error by splitting our data\!

<br> <br>

<center>

<img src="/module3/splitted.png"  width = "100%" alt="404 image" />

</center>

Notes:

We keep aside some randomly selected portion from the training data.

We `fit` (train) a model on the training portion only.

We `score` (assess) the trained model on this set-aside data to get a
sense of how well the model would be able to generalize.

We pretend that the kept aside data is representative of the real
distribution (𝐷) of data.

---

## Simple train and test split

<br> <br>

<center>

<img src="/module3/train-test-split.png"  width = "100%" alt="404 image" />

</center>

Notes:

The data is shuffled before splitting.

We then split up our data into 2 separate sections.

The picture shows an 80%-20% split of a toy dataset with 10 examples.

Usually, when we do machine learning we split the data before doing
anything and put the test data in an imaginary chest lock.

---

<center>

<img src="/module3/split_funct.png"  width = "80%" alt="404 image" />

</center>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">*Attribution*</a>

Notes:

We can do this very easily with a tool from Scikit Learn.

It gives us the versatility to either pass `X` and `y` or a dataframe
with both `X` and `y` in it.

It uses useful arguments such as:

  - `test_size`
  - `train_size`
  - `random_state`

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
cities_df
```

```out
     longitude  latitude country
0    -130.0437   55.9773     USA
1    -134.4197   58.3019     USA
2    -123.0780   48.9854     USA
3    -122.7436   48.9881     USA
4    -122.2691   48.9951     USA
..         ...       ...     ...
204   -72.7218   45.3990  Canada
205   -66.6458   45.9664  Canada
206   -79.2506   42.9931  Canada
207   -72.9406   45.6275  Canada
208   -79.4608   46.3092  Canada

[209 rows x 3 columns]
```

Notes: Let’s test it out in action with the Canadian and United States
cities data that we saw in module 2.

---

``` python
X = cities_df.drop(["country"], axis=1)
X
```

```out
     longitude  latitude
0    -130.0437   55.9773
1    -134.4197   58.3019
2    -123.0780   48.9854
3    -122.7436   48.9881
4    -122.2691   48.9951
..         ...       ...
204   -72.7218   45.3990
205   -66.6458   45.9664
206   -79.2506   42.9931
207   -72.9406   45.6275
208   -79.4608   46.3092

[209 rows x 2 columns]
```

Notes: First we have our `X` dataframe.

---

``` python
y = cities_df["country"]
y
```

```out
0         USA
1         USA
2         USA
3         USA
4         USA
        ...  
204    Canada
205    Canada
206    Canada
207    Canada
208    Canada
Name: country, Length: 209, dtype: object
```

Notes: Followed by our `y`, target column.

---

``` python
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
```

``` python
X_train.head(3)
```

```out
     longitude  latitude
160   -76.4813   44.2307
127   -81.2496   42.9837
169   -66.0580   45.2788
```

Notes:

First we import `train_test_split` from `sklearn.model_selection`.

We split our data and separate our `X` and `y` variables into 4 separate
objects that we name:

  - `X_train`
  - `X_test`
  - `y_train`
  - `y_test`

---

``` python
X_test.head(3)
```

```out
     longitude  latitude
172   -64.8001   46.0980
175   -82.4066   42.9746
181  -111.3885   56.7292
```

``` python
y_train.head(3)
```

```out
160    Canada
127    Canada
169    Canada
Name: country, dtype: object
```

``` python
y_test.head(3)
```

```out
172    Canada
175    Canada
181    Canada
Name: country, dtype: object
```

Notes:

<br>

---

``` python
shape_dict = {"Data portion": ["X", "y", "X_train", "y_train", "X_test", "y_test"],
    "Shape": [X.shape, y.shape,
              X_train.shape, y_train.shape,
              X_test.shape, y_test.shape]}

shape_df = pd.DataFrame(shape_dict)
shape_df
```

```out
  Data portion     Shape
0            X  (209, 2)
1            y    (209,)
2      X_train  (167, 2)
3      y_train    (167,)
4       X_test   (42, 2)
5       y_test     (42,)
```

Notes:

Let’s take a look at the shape of each of these dataframes now.

---

``` python
train_df, test_df = train_test_split(cities_df, test_size = 0.2, random_state = 123)

X_train, y_train = train_df.drop(["country"], axis=1), train_df["country"]

X_test, y_test = test_df.drop(["country"], axis=1), test_df["country"]

train_df.head()
```

```out
     longitude  latitude country
160   -76.4813   44.2307  Canada
127   -81.2496   42.9837  Canada
169   -66.0580   45.2788  Canada
188   -73.2533   45.3057  Canada
187   -67.9245   47.1652  Canada
```

Notes: Sometimes we may want to keep the target in the train split for
EDA or for visualization.

That’s not a problem.

We can do this by splitting our `cities_df` dataframe into a train and
test dataframe as objects `train_df`, `test_df`, and then separate the
features from the target after the fact.

---

``` python
chart_cities = alt.Chart(train_df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue'])))
chart_cities
```
<img src="/module3/chart_cities.png" alt="A caption" width="63%" />

Notes: Now we can plot the data from the training data `train_df`, we
can differentiate between the Canadian cities (red) and the United
States cities (blue).

---

``` python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

```out
DecisionTreeClassifier()
```

Notes:

We can build our model and fit our data.

---

<center>

<img src="/module3/module3_01a.png"  width = "83%" alt="404 image" />

</center>

Notes:

---

``` python
print("Train error: " + str(round(1 - model.score(X_train, y_train), 2)))
```

```out
Train error: 0.0
```

``` python
print("Test error: " + str(round(1 - model.score(X_test, y_test), 2)))
```

```out
Test error: 0.29
```

Notes:

Let’s examine the train and test accuracies with the split now.

Now when we examine the train and test error with the split we can see
our model does not do as well a job generalizing as we did before.

Our training error is 0, however, our testing error is 0.26.

---

<img src="/module3/module3_01/unnamed-chunk-17-1.png" width="105%" />

Notes:

The plot above shows the boundaries from the tree trained on training
data and the test data.

---

### *test\_size* and *train\_size* arguments

``` python
train_df, test_df = train_test_split(cities_df, test_size = 0.2, random_state = 123)
```

``` python
shape_dict2 = {"Data portion": ["cities_df", "train_df", "test_df"],
    "Shape": [cities_df.shape, train_df.shape,
              test_df.shape]}

shape_df2 = pd.DataFrame(shape_dict2)
shape_df2
```

```out
  Data portion     Shape
0    cities_df  (209, 3)
1     train_df  (167, 3)
2      test_df   (42, 3)
```

Notes:

Let’s take a closer look at the arguments in the splitting tool.

When we specify how we want to split the data, we can specify either
`test_size` or `train_size`.

See the documentation
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">here</a>.

There is no hard and fast rule on the split sizes should we use and it
depends upon how much data is available to us

Some common splits are 90/10, 80/20, 70/30 (training/test).

In the above example, we used an 80/20 split.

---

### *random\_state* argument

``` python
train_df_rs5, test_df_rs5 = train_test_split(cities_df, test_size = 0.2, random_state = 5)
```

``` python
train_df_rs7, test_df_rs7 = train_test_split(cities_df, test_size = 0.2, random_state = 7)
```

``` python
train_df_rs5.head(3)
```

```out
    longitude  latitude country
39   -96.7969   32.7763     USA
55   -97.5171   35.4730     USA
40  -121.8906   37.3362     USA
```

``` python
train_df_rs7.head(3)
```

```out
     longitude  latitude country
128  -118.7148   50.4165  Canada
195  -122.7454   53.9129  Canada
99    -72.0968   45.0072  Canada
```

Notes:

The data is shuffled before splitting which is a crucial step. The
`random_state` argument controls this shuffling.

In the example above we used `random_state=5` and `random_state=7` and
we can see that they contain different observations.

Setting the random\_state is useful when we want reproducible results.

---

# Let’s apply what we learned\!

Notes: <br>

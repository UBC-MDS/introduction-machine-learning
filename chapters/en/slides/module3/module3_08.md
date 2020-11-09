---
type: slides
---

# Cross-validation

Notes: <br>

---

## Single split problems

<br> <br> <br>

<center>

<img src="/module3/train-valid-test-split.png"  width = "100%" alt="404 image" />

</center>

Notes:

We saw that it’s necessary to split our data into multiple different
sets/splits but is having a single train and validation split optimal?

The problem with having a single train/validation split is that now we
are using only a portion of our data for training and only a portion for
validation.

If our dataset is small we might end up with a tiny training and/or
validation set.

We might also be unlucky with our splits such that they don’t align well
or don’t well represent our test data.

---

## So what do we do?

### 𝑘-fold cross-validation

<center>

<img src="/module3/cross-validation.png"  width = "100%" alt="404 image" />

</center>

Notes:

We use something called ***cross-validation*** or ***𝑘-fold
cross-validation*** as a solution to this problem..

Cross-validation helps us use all of our data for training/validation\!

Cross-validation consists of splitting the data into k-folds ( 𝑘\>2 ,
often 𝑘=10 ). In the picture below 𝑘=4 .

Each “fold” gets a turn at being the validation set.

Each fold gives a score and we usually average our 𝑘 results.

It’s better to notice the variation in the scores across folds.

We can get a more “robust” score on unseen data.

The main disadvantage here is that this is slower, which is a problem
for bigger datasets / more complex models.

---

## Cross-validation using scikit-learn

``` python
df = pd.read_csv("data/canada_usa_cities.csv")
X = df.drop(columns=["country"])
y = df["country"]
```

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
```

Notes:

let’s bring in our Canadian/United States cities data and split it.

---

``` python
from sklearn.model_selection import cross_val_score

model = DecisionTreeClassifier(max_depth=4)
cv_score = cross_val_score(model, X_train, y_train, cv=5)
cv_score
```

```out
array([0.76470588, 0.82352941, 0.78787879, 0.78787879, 0.84848485])
```

Notes:

First we import `cross_val_score` from `sklearn.model_selection`.

We create our decision tree model.

We use `cross_val_score()` and specify the model and the training
features and target as arguments. We also specify `cv` which determines
the cross-validation splitting strategy or how many “folds” there are.

Here we are saying there at 5 folds on the data.

In each fold, it fits the model on the training portion and scores on
the validation portion.

We can see the output is a list of validation scores in each fold.

---

``` python
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
cv_scores
```

```out
array([0.76470588, 0.82352941, 0.70588235, 0.94117647, 0.82352941, 0.82352941, 0.70588235, 0.9375    , 0.9375    , 0.9375    ])
```

``` python
cv_scores.mean()
```

```out
0.8400735294117647
```

Notes:

We can change the number of folds too. Now, when we change it to 10, we
get 10 different scores.

---

``` python
from sklearn.model_selection import cross_validate
```

``` python
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
scores
```

```out
{'fit_time': array([0.00217772, 0.003093  , 0.00193882, 0.00194216, 0.00184488, 0.00194931, 0.00215077, 0.00201583, 0.00187898, 0.00199509]), 'score_time': array([0.00143719, 0.002388  , 0.00158   , 0.00139189, 0.00148988, 0.0016458 , 0.00148416, 0.0020113 , 0.00169086, 0.00140786]), 'test_score': array([0.76470588, 0.82352941, 0.70588235, 0.94117647, 0.82352941, 0.82352941, 0.70588235, 0.9375    , 0.9375    , 0.9375    ]), 'train_score': array([0.91333333, 0.90666667, 0.90666667, 0.9       , 0.90666667, 0.91333333, 0.92      , 0.90066225, 0.90066225, 0.90066225])}
```

Notes:

`cross_val_score()` is the simpler scikit-learn function for
cross-validation.

Let us access training and validation scores.

---

``` python
scores
```

```out
{'fit_time': array([0.00217772, 0.003093  , 0.00193882, 0.00194216, 0.00184488, 0.00194931, 0.00215077, 0.00201583, 0.00187898, 0.00199509]), 'score_time': array([0.00143719, 0.002388  , 0.00158   , 0.00139189, 0.00148988, 0.0016458 , 0.00148416, 0.0020113 , 0.00169086, 0.00140786]), 'test_score': array([0.76470588, 0.82352941, 0.70588235, 0.94117647, 0.82352941, 0.82352941, 0.70588235, 0.9375    , 0.9375    , 0.9375    ]), 'train_score': array([0.91333333, 0.90666667, 0.90666667, 0.9       , 0.90666667, 0.91333333, 0.92      , 0.90066225, 0.90066225, 0.90066225])}
```

``` python
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002178    0.001437    0.764706     0.913333
1  0.003093    0.002388    0.823529     0.906667
2  0.001939    0.001580    0.705882     0.906667
3  0.001942    0.001392    0.941176     0.900000
4  0.001845    0.001490    0.823529     0.906667
5  0.001949    0.001646    0.823529     0.913333
6  0.002151    0.001484    0.705882     0.920000
7  0.002016    0.002011    0.937500     0.900662
8  0.001879    0.001691    0.937500     0.900662
9  0.001995    0.001408    0.937500     0.900662
```

Notes:

`scores` is returned as a dictionary but it’s much easier to understand
if we convert it to a dataframe.

---

``` python
pd.DataFrame(scores).mean()
```

```out
fit_time       0.002099
score_time     0.001653
test_score     0.840074
train_score    0.906865
dtype: float64
```

``` python
cross_val_score(model, X_train, y_train, cv=10).mean()
```

```out
0.8400735294117647
```

``` python
pd.DataFrame(scores).std()
```

```out
fit_time       0.000365
score_time     0.000317
test_score     0.094993
train_score    0.006822
dtype: float64
```

Notes:

Using `cross_validate()` instead of `cross_val_score()` gives us more
information.

`cross_val_score()` was just returning that last column, but here we get
the time spent and the training scores.

We can calculate the mean of each column on the 10 folds.

It’s a bit unfortunate that they call it “test\_score” in scikit-learn;
for us this is a validation score.

We can see the mean from this is similar to the mean from
`cross_val_score()`.

Normally we calculate the mean cross-validation score but sometimes it
would be useful to look at the range and standar deviation of the folds.

---

### Our typical supervised learning set up is as follows:

<br>

1.  Given training data with `X` and `y`.
2.  We split our data into `X_train, y_train, X_test, y_test`.
3.  Hyperparameter optimization using cross-validation on `X_train` and
    `y_train`.
4.  We assess the best model using `X_test` and `y_test`.
5.  The **test score** tells us how well our model generalizes.
6.  If the **test score** is reasonable, we deploy the model.

Notes: We are given training data with features `X` and target `y`.

We split the data into train and test portions: `X_train, y_train,
X_test, y_test`.

We carry out hyperparameter optimization using cross-validation on the
train portion: `X_train` and `y_train`.

We assess our best performing model on the test portion: `X_test` and
`y_test`.  
What we care about is the **test score**, which tells us how well our
model can be generalized.

If this test score is “reasonable” we deploy the model which will be
used on new unseen examples.

How do we know whether this test score is reasonable?

We will discuss this in the next section\!

---

# Let’s apply what we learned\!

Notes: <br>

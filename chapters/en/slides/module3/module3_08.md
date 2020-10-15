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

We can get a more “robust” measure of error on unseen data.

The main disadvantage here is that this is slower, which is a problem
for bigger datasets / more complex models.

---

## Cross-validation using scikit-learn

``` python
df = pd.read_csv("data/canada_usa_cities.csv")
X = df.drop(["country"], axis=1)
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
cross_val_score(model, X_train, y_train, cv=10)
```

```out
array([0.76470588, 0.82352941, 0.70588235, 0.94117647, 0.82352941, 0.82352941, 0.70588235, 0.9375    , 0.9375    , 0.9375    ])
```

Notes:

We can change the number of folds too. Now, when we change it to 10, we
get 10 different scores.

---

``` python
from sklearn.model_selection import cross_validate
```

``` python
scores = cross_validate(model, X, y, cv=10, return_train_score=True)
scores
```

```out
{'fit_time': array([0.00201392, 0.00191522, 0.00260496, 0.00279403, 0.00188708, 0.00197005, 0.00221491, 0.00188708, 0.00195885, 0.00199604]), 'score_time': array([0.00185394, 0.00140381, 0.002455  , 0.00145411, 0.00203991, 0.00141191, 0.00143313, 0.00283217, 0.00213909, 0.00141811]), 'test_score': array([0.57142857, 0.38095238, 0.42857143, 0.66666667, 1.        , 1.        , 0.85714286, 0.95238095, 1.        , 0.9       ]), 'train_score': array([0.89361702, 0.88829787, 0.93617021, 0.92021277, 0.86702128, 0.86702128, 0.88829787, 0.87234043, 0.86702128, 0.88888889])}
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
{'fit_time': array([0.00201392, 0.00191522, 0.00260496, 0.00279403, 0.00188708, 0.00197005, 0.00221491, 0.00188708, 0.00195885, 0.00199604]), 'score_time': array([0.00185394, 0.00140381, 0.002455  , 0.00145411, 0.00203991, 0.00141191, 0.00143313, 0.00283217, 0.00213909, 0.00141811]), 'test_score': array([0.57142857, 0.38095238, 0.42857143, 0.66666667, 1.        , 1.        , 0.85714286, 0.95238095, 1.        , 0.9       ]), 'train_score': array([0.89361702, 0.88829787, 0.93617021, 0.92021277, 0.86702128, 0.86702128, 0.88829787, 0.87234043, 0.86702128, 0.88888889])}
```

``` python
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002014    0.001854    0.571429     0.893617
1  0.001915    0.001404    0.380952     0.888298
2  0.002605    0.002455    0.428571     0.936170
3  0.002794    0.001454    0.666667     0.920213
4  0.001887    0.002040    1.000000     0.867021
5  0.001970    0.001412    1.000000     0.867021
6  0.002215    0.001433    0.857143     0.888298
7  0.001887    0.002832    0.952381     0.872340
8  0.001959    0.002139    1.000000     0.867021
9  0.001996    0.001418    0.900000     0.888889
```

Notes:

`scores` is returned as a dictionary but it’s much easier to understand
if we convert it to a dataframe.

---

``` python
pd.DataFrame(pd.DataFrame(scores).mean())
```

```out
                    0
fit_time     0.002124
score_time   0.001844
test_score   0.775714
train_score  0.888889
```

Notes:

Using `cross_validate()` instead of `cross_val_score()` gives us more
information.

`cross_val_score()` was just returning that last column, but here we get
the time spent and the training scores.

We can calculate the mean of each column on the 10 folds.

It’s a bit unfortunate that they call it “test\_score” in scikit-learn;
for us this is a validation score.

---

### Our typical supervised learning set up is as follows:

<br>

1.  Given training data with `X` and `y`.
2.  We split our data into `X_train, y_train, X_test, y_test`.
3.  Hyperparameter optimization using cross-validation on `X_train` and
    `y_train`.
4.  We assess the best model using `X_test` and `y_test`.
5.  The **test error** tells us how well our model generalizes.
6.  If the **test error** is reasonable, we deploy the model.

Notes: We are given training data with features `X` and target `y`.

We split the data into train and test portions: `X_train, y_train,
X_test, y_test`.

We carry out hyperparameter optimization using cross-validation on the
train portion: `X_train` and `y_train`.

We assess our best performing model on the test portion: `X_test` and
`y_test`.  
What we care about is the **test error**, which tells us how well our
model can be generalized.

If this test error is “reasonable” we deploy the model which will be
used on new unseen examples.

How do we know whether this test error is reasonable?

We will discuss this in the next section\!

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Cross validation

Notes: <br>

---

## Single split problems

<br>

<center>

<img src="/module3/train-valid-test-split.png"  width = "100%" alt="404 image" />

</center>

Notes:

We saw that it’s necessary to split our data into multiple different
sets/splits but is having a single train and validation split optimal?

The problem with having a single train/validation split is that now we
are using only a portion of our data for training and onlu a portion for
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

Cross Validation helps us use all of our data for training/validation\!

Cross-validation consists of splitting the data into k-folds ( 𝑘\>2 ,
often 𝑘=10 ). In the picture below 𝑘=4 .

Each “fold” gets a turn at being the validation set.

Each fold gives a score and we usually average our 𝑘 results.

It’s better to notice the variation in the scores across folds.

We can get a more “robust” measure of error on unseen data.

The main disadvantage here is that this is slower, which is a problem
for bigger data sets / more complex models.

---

## Cross validation using scikit-learn

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
get 10 different scores

---

``` python
from sklearn.model_selection import cross_validate
```

``` python
scores = cross_validate(model, X, y, cv=10, return_train_score=True)
scores
```

```out
{'fit_time': array([0.00220394, 0.00187182, 0.00186992, 0.00209212, 0.00202441, 0.00237799, 0.0022459 , 0.00207806, 0.00216293, 0.00210214]), 'score_time': array([0.00145507, 0.00139809, 0.00212693, 0.00145793, 0.00181079, 0.00168109, 0.00144506, 0.00144625, 0.00141811, 0.00159478]), 'test_score': array([0.57142857, 0.38095238, 0.42857143, 0.66666667, 1.        , 1.        , 0.85714286, 0.9047619 , 1.        , 0.9       ]), 'train_score': array([0.89361702, 0.88829787, 0.93617021, 0.92021277, 0.86702128, 0.86702128, 0.88829787, 0.87234043, 0.86702128, 0.88888889])}
```

Notes:

`cross_val_score()` is the simpler scikit-learn function for
cross-validation.

Let’s us access training and validation scores.

`scores` is returned as a dictionary but it’s much easier to understand
if we convert it to a dataframe.

---

``` python
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002204    0.001455    0.571429     0.893617
1  0.001872    0.001398    0.380952     0.888298
2  0.001870    0.002127    0.428571     0.936170
3  0.002092    0.001458    0.666667     0.920213
4  0.002024    0.001811    1.000000     0.867021
5  0.002378    0.001681    1.000000     0.867021
6  0.002246    0.001445    0.857143     0.888298
7  0.002078    0.001446    0.904762     0.872340
8  0.002163    0.001418    1.000000     0.867021
9  0.002102    0.001595    0.900000     0.888889
```

``` python
pd.DataFrame(pd.DataFrame(scores).mean())
```

```out
                    0
fit_time     0.002103
score_time   0.001583
test_score   0.770952
train_score  0.888889
```

Notes:

Using `cross_validate()` instead of `cross_val_score()` gives us more
information.

`cross_val_score()` was just returning that last column, but here we get
the time spent and the training scores.

We can calculate the mean of each column for the total 10 folds.

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

How do we know whether this test error is reasonable? We will discuss
this in the next section\!

---

# Let’s apply what we learned\!

Notes: <br>

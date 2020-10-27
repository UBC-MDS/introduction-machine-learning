---
type: slides
---

# The importance of preprocessing

Notes: <br>

---

<br> <br>

### So far …

  - Models: Decision trees, 𝑘-NNs, SVMs with RBF kernel.
  - Fundamentals: Train-validation-test split, cross-validation, the
    fundamental tradeoff, the golden rule.

<br> <br>

### Now

**Preprocessing**: Transforming input data into a format a machine
learning model can use and understand.

Notes:

So far we have seen:

  - Three ML models (decision trees, 𝑘-NNs, SVMs with RBF kernel)
  - ML fundamentals (train-validation-test split, cross-validation, the
    fundamental tradeoff, the golden rule)

Are we ready to do machine learning on real-world datasets? - Very often
real-world datasets need to be transformed or ***preprocessed*** before
we use them to build ML models.

---

## Basketball dataset

``` python
bball_df = pd.read_csv('data/bball.csv')
bball_df.head()
```

```out
               full_name  rating jersey                  team position     b_day  height  weight      salary country  draft_year draft_round draft_peak          college
0           LeBron James      97    #23    Los Angeles Lakers        F  12/30/84    2.06   113.4  37436858.0     USA        2003           1          1              NaN
1          Kawhi Leonard      97     #2  Los Angeles Clippers        F  06/29/91    2.01   102.1  32742000.0     USA        2011           1         15  San Diego State
2  Giannis Antetokounmpo      96    #34       Milwaukee Bucks      F-G  12/06/94    2.11   109.8  25842697.0  Greece        2013           1         15              NaN
3           Kevin Durant      96     #7         Brooklyn Nets        F  09/29/88    2.08   104.3  37199000.0     USA        2007           1          2            Texas
4           James Harden      96    #13       Houston Rockets        G  08/26/89    1.96    99.8  38199000.0     USA        2009           1          3    Arizona State
```

``` python
bball_df = bball_df[(bball_df['position'] =='G') | (bball_df['position'] =='F')]
X = bball_df[['weight', 'height', 'salary']]
y =bball_df["position"]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, random_state=123)
```

``` python
X_train.head()
```

```out
     weight  height     salary
152    79.4    1.88  1588231.0
337    82.1    1.91  2149560.0
130   106.6    2.03  6500000.0
340   106.1    2.08  2961120.0
50     96.2    1.93  4861207.0
```

Notes:

In module 3, we used a portion of the basketball dataset to predict a
players position using `DecisionTreeClassifier`.

Can we use 𝑘-NN classifier for this task?

Intuition: To predict whether a particular player is a pointguard (‘G’)
or a forward (‘F’) (query point)

  - Find the players that are closest to the query point
  - Let them vote on the target
  - Take the majority vote as the target for the query point

---

### Geometric view of tabular data and dimensions

``` python
dummy = DummyClassifier(strategy="most_frequent")
scores = cross_validate(dummy, X_train, y_train, return_train_score=True)
print('Mean validation score', scores['test_score'].mean().round(2))
```

```out
Mean validation score 0.57
```

``` python
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.000736    0.000535    0.571429     0.571429
1  0.000523    0.000345    0.571429     0.571429
2  0.000500    0.000335    0.571429     0.571429
3  0.000537    0.000341    0.571429     0.571429
4  0.000870    0.000501    0.571429     0.571429
```

``` python
knn = KNeighborsClassifier()
scores = cross_validate(knn, X_train, y_train, return_train_score=True)
print('Mean validation score', scores['test_score'].mean().round(2))
```

```out
Mean validation score 0.5
```

``` python
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002476    0.003965    0.489796     0.688776
1  0.002257    0.003342    0.530612     0.678571
2  0.003239    0.003775    0.530612     0.698980
3  0.002449    0.003501    0.530612     0.704082
4  0.002109    0.003740    0.408163     0.724490
```

Notes:

First, let’s see what scores we get if we simply predict the most
occuring position in the dataset using our dummy classifier.

Now if we build our 𝑘-NN classifier we determine that it gets even
*worse* scores\! Why?

---

``` python
two_players = X_train.sample(2, random_state=42)
two_players
```

```out
     weight  height     salary
285    91.2    1.98  1882867.0
236   112.0    2.08  2000000.0
```

``` python
euclidean_distances(two_players)
```

```out
array([[     0.        , 117133.00184683],
       [117133.00184683,      0.        ]])
```

``` python
two_players_subset = two_players[["salary"]]
two_players_subset
```

```out
        salary
285  1882867.0
236  2000000.0
```

``` python
euclidean_distances(two_players_subset)
```

```out
array([[     0., 117133.],
       [117133.,      0.]])
```

Notes:

Let’s have a look at just 2 players as calculate the distance between
them.

We can see the distance between player 285 and 236 is 117133.00184683.

What happens if we only consider the `salary` column though?

It looks like we get almost the same distance\!

The distance is completely dominated by the the features with larger
values.

The features with smaller values are being ignored.

Does it matter?

  - Yes\! Scale is based on how data was collected.
  - Features on a smaller scale can be highly informative and there is
    no good reason to ignore them.
  - We want our model to be robust and not sensitive to the scale.

Was this a problem for decision trees?

  - No. In decision trees we ask questions on one feature at a time.

So what do we do about this?

Well, we have to scale the columns they they are all using a similar
range of values\!

Luckily Sklearn has tools for this.

---

## Scaling

| Approach      | What it does          | How to update 𝑋 (but see below\!)                  | sklearn implementation                                                                                         |
| ------------- | --------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Normalization | sets range to \[0,1\] | `X -= np.min(X,axis=0)`<br>`X /= np.max(X,axis=0)` | [`MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), |
| "             | Standardization       | sets sample mean to 0, s.d. to 1                   | `X -= np.mean(X,axis=0)`<br>`X /=  np.std(X,axis=0)`                                                           |

There are all sorts of articles on this; see,
e.g. [here](http://www.dataminingblog.com/standardization-vs-normalization/)
and
[here](https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc).

Notes:

There are different ways to scales.

2 popular options are called Normalization and Standardization. We are
not going to explain in detail what is going on behind these tools but
more so on how to implement them.

---

## `scikit-learn`’s [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

``` python
from sklearn.preprocessing import StandardScaler
```

<br>

``` python
scaler = StandardScaler()    # Create feature transformer object
scaler.fit(X_train) # fitting the transformer on the train split 
```

```out
StandardScaler()
```

``` python
X_train_scaled = scaler.transform(X_train) # transforming the train split
X_test_scaled = scaler.transform(X_test) # transforming the test split
pd.DataFrame(X_train_scaled, columns = X_train.columns).head()
```

```out
     weight    height    salary
0 -1.552775 -1.236056 -0.728809
1 -1.257147 -0.800950 -0.670086
2  1.425407  0.939473 -0.214967
3  1.370661  1.664650 -0.585185
4  0.286690 -0.510879 -0.386408
```

Notes:

We’ll be concentrating on `scikit-learn`’s
[`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
which is a `transformer`.

For now, try to only focus on the syntax.

We’ll talk about scaling in a bit.

1.  Create feature transformer object. this is done in a similar way to
    how we create a model.
2.  Fitting the transformer on the train split
3.  Transform the train split using `.transform()`
4.  Then tranform the test split.

<!-- end list -->

  - `sklearn` uses `fit` and `transform` paradigms for feature
    transformations. (In model building it was `fit` and `predict` or
    `score`)
  - We `fit` the transformer on the train split and then `transform` the
    train split as well as the test split.
  - We apply the same transformations on the test split.

---

## Scikit learn’s *predict* vs *transform*

``` python
model.fit(X_train, y_train)
X_train_predictions = model.predict(X_train)
X_test_predictions = model.predict(X_test)
```

``` python
transformer.fit(X_train, [y_train])
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

``` python
transformer.fit_transform(X_train)
```

Notes:

Suppose we have a named `model` which is either a classification or
regression model.

We can compare it with `transformer` which is a transformer used to
change the input representation like to scales numeric features.

You can pass `y_train` in `fit` but it’s usually ignored. It allows you
to pass it just to be consistent with usual usage of `sklearn`’s `fit`
method.

You can also carry out fitting and transforming in one call using
`fit_transform`. But be mindful to use it only on the train split and
**not** on the test split.

---

``` python
knn_unscaled = KNeighborsClassifier();
knn_unscaled.fit(X_train, y_train)
```

```out
KNeighborsClassifier()
```

``` python
print('Train score: %0.3f' %(knn_unscaled.score(X_train, y_train)))
```

```out
Train score: 0.706
```

``` python
print('Test score: %0.3f' %(knn_unscaled.score(X_test, y_test)))
```

```out
Test score: 0.452
```

``` python
knn_scaled = KNeighborsClassifier();
knn_scaled.fit(X_train_scaled, y_train)
```

```out
KNeighborsClassifier()
```

``` python
print('Train score: %0.3f' %(knn_scaled.score(X_train_scaled, y_train)))
```

```out
Train score: 0.939
```

``` python
print('Test score: %0.3f' %(knn_scaled.score(X_test_scaled, y_test)))
```

```out
Test score: 0.887
```

Notes:

Do you expect `DummyClassifier` results to change after scaling the
data?

Let’s check whether scaling makes any difference for 𝑘-NNs.

The scores with scaled data are better compared to the unscaled data in
case of 𝑘-NNs.

We am not carrying out cross-validation here for a reason that that
we’ll look into soon.

We are being a bit sloppy here by using the test set several times for
teaching purposes.

But when you build an ML pipeline, please do assessment on the test set
only once.

---

# Let’s apply what we learned\!

Notes: <br>

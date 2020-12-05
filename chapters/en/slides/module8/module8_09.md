---
type: slides
---

# Logistic regression

Notes: <br>

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
X_train, y_train = train_df.drop(columns=["country"], axis=1), train_df["country"]
X_test, y_test = test_df.drop(columns=["country"], axis=1), test_df["country"]

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

Notes:

Next, we are going to introduce to you a new model called **logistic
regression**.

It’s very similar to the `Ridge` classifier we sae earlier but this one
has some key differences.

For one, we can use it with classification instead of regression
problems.

For that reason we are going to bring back our cities dataset we saw at
the beginning of this course.

---

## Setting the stage

``` python
from sklearn.dummy import DummyClassifier
dc = DummyClassifier(strategy="prior")

scores= pd.DataFrame(cross_validate(dc, X_train, y_train, return_train_score=True))
scores
```

```out
   fit_time  score_time  test_score  train_score
0  0.000708    0.000482    0.588235     0.601504
1  0.000627    0.000396    0.588235     0.601504
2  0.000553    0.000432    0.606061     0.597015
3  0.000516    0.000363    0.606061     0.597015
4  0.000902    0.000753    0.606061     0.597015
```

Notes:

Although we don’t always do this in the slides, we should always be
building a baseline model before we do any type of meaningful modelling.

Let’s do that before we get straight into it.

Now we can have a better idea on how well our model performs.

---

``` python
from sklearn.linear_model import LogisticRegression
```

``` python
lr = LogisticRegression()
scores = pd.DataFrame(cross_validate(lr, X_train, y_train, return_train_score=True))
scores
```

```out
   fit_time  score_time  test_score  train_score
0  0.021131    0.002766    0.852941     0.827068
1  0.013887    0.001981    0.823529     0.827068
2  0.013444    0.001905    0.696970     0.858209
3  0.014567    0.002652    0.787879     0.843284
4  0.013251    0.001736    0.939394     0.805970
```

Notes:

We import `LogisticRegression` from the `sklearn.linear_model` library
like we did with `Ridge`.

This time we can see that our training and cross-validation scores have
increased from those of our `DummyClassifier`.

---

## Visualizing our model

<img src="/module8/module8_09/unnamed-chunk-6-1.png" width="80%" style="display: block; margin: auto;" />

Notes:

We saw that with SVMs and decision trees we could visualize our model
with decision boundaries and we can do the same thing here.

Here we can see we get a line that separates our two target classes.

---

<br> <br> <br> <br>

<center>

<img src="/module8/triple_graph.png"  width = "100%" alt="404 image" />

</center>

Notes:

If we look at some other models that we did this in comparison for you
can understand a bit more on why we call Logistic Regression a “linear
Classifiers”.

Notice a linear decision boundary (a line in our case).

---

# Coefficients

``` python
lr = LogisticRegression()
lr.fit(X_train, y_train); 
```

``` python
print("Model weights:", lr.coef_)
print("Model intercept:", lr.intercept_)
```

``` out
Model weights: [[-0.04108149 -0.33683126]]
Model intercept: [10.8869838]
```

``` python
data = {'features': X_train.columns, 'coefficients':lr.coef_[0]}
pd.DataFrame(data)
```

```out
    features  coefficients
0  longitude     -0.041081
1   latitude     -0.336831
```

Notes:

Just like we saw for `Ridge`. we can get the equation of that line and
the coefficients of our `latitude` and `longitude` features using
`.coef_`.

In this case we see that both are negative weights,

We also can see that the weight of latitude is larger in magnitude.

This makes a lot of sense because Canada as a country lies above the USA
and so we expect `latitude` values to contribute more to a prediction
than `longitude` which Canada and the `USA` have quite similar values.

---

## Predictions

``` python
example = X_test.iloc[0,:]
example.tolist()
```

```out
[-64.8001, 46.098]
```

``` python
lr.intercept_ + (example.tolist() * lr.coef_).sum(axis=1)
```

```out
array([-1.97817876])
```

``` python
lr.predict([example])
```

```out
array(['Canada'], dtype=object)
```

Notes:

Again let’s take an example from our test and calculate the outcome
using our weights and intercept.

We get a value of -1.978.

In `Ridge` our prediction would be the calculated result so -1.97, but
for logistic regression we check the **sign** of the calculation only.

Our threshold is 0. - If the result was positive, it predicts the one
class; if negative, it predict the other.

That means everything negative corresponds to “Canada” and everything
positive predicts a class of “USA”.

If we use `predict`, it gives us the same result as well\!

These are “hard predictions” but we can also use this for something
called “soft predictions” as well (That’s in the next slide deck\!).

---

## Hyperparameter: C (A new one)

``` python
scores_dict ={
"C" :10.0**np.arange(-6,2,1),
"train_score" : list(),
"cv_score" : list(),
}
for C in scores_dict['C']:
    lr_model = LogisticRegression(C=C)
    results = cross_validate(lr_model, X_train, y_train, return_train_score=True)
    scores_dict['train_score'].append(results["train_score"].mean())
    scores_dict['cv_score'].append(results["test_score"].mean())
```

``` python
pd.DataFrame(scores_dict)
```

```out
           C  train_score  cv_score
0   0.000001     0.598810  0.598930
1   0.000010     0.598810  0.598930
2   0.000100     0.664707  0.658645
3   0.001000     0.784424  0.790731
4   0.010000     0.827842  0.826203
5   0.100000     0.832320  0.820143
6   1.000000     0.832320  0.820143
7  10.000000     0.832320  0.820143
```

Notes:

At this point you should be feeling pretty comfortable with
hyperparameters.

We saw that `Ridge` has the hyperparameter `alpha`, well `C`
(annoyingly) has the opposite effect on the fundamental trade off.

In general, we say smaller `C` leads to a less complex model (whereas
with `Ridge`, lower `alpha` means higher complexity).

higher values of `C` leads to more overfitting and lower values to less
overfitting.

---

``` python
param_grid = {
    "C": scipy.stats.uniform(0, 100)}

lr = LogisticRegression()
grid_search = RandomizedSearchCV(lr, param_grid, cv=5, return_train_score=True, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 10 candidates, totalling 50 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.9s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    1.9s finished
```

``` python
grid_search.best_params_
```

```out
{'C': 12.786946328668414}
```

``` python
grid_search.best_score_
```

```out
0.8201426024955436
```

`LogisticRegression`’s default `C` hyperparameter is 1.

Let’s see what kind of value we get if we do `RandomizedGrid`.

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Automated hyperparameter optimization

Notes: <br>

---

## The problem with hyperparameters

  - We may have a lot of them.
  - Picking reasonable hyperparameters is important -\> it helps avoid
    underfit or overfit models.
  - Nobody knows exactly how to choose them.
  - May interact with each other in unexpected ways.
  - The best settings depend on the specific data/problem.
  - Can take a long time to execute.

Notes: <br>

---

## How to pick hyperparameters

<br>

### Manual hyperparameter optimization

**Advantages**:

  - We may have some intuition about what might work.

**Disadvantages**:

  - It takes a lot of work.  
  - In some cases, intuition might be worse than a data-driven approach.

### Automated hyperparameter optimization

**Advantages**:

  - Reduce human effort.  
  - Less prone to error.  
  - Data-driven approaches may be effective.

**Disadvantages**:

  - It may be hard to incorporate intuition.  
  - Overfitting on the validation set.

Notes:

<br>

---

### Automated hyperparameter optimization

<br> <br>

  - Exhaustive grid search:
    <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html" target="_blank">`sklearn.model_selection.GridSearchCV`</a>  
  - Randomized hyperparameter optimization:
    <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html" target="_blank">`sklearn.model_selection.RandomizedSearchCV`</a>

Notes:

There are two automated hyperparameter search methods in scikit-learn.

The “CV” stands for cross-validation; these searchers have
cross-validation built right in.

---

## Bring in the data

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
X_train, y_train = train_df.drop(columns=['country']), train_df['country']
X_test, y_test = test_df.drop(columns=['country']), test_df['country']
X_train.head()
```

```out
     longitude  latitude
160   -76.4813   44.2307
127   -81.2496   42.9837
169   -66.0580   45.2788
188   -73.2533   45.3057
187   -67.9245   47.1652
```

Notes:

Let’s bring back the cities dataset we worked with in the last model.

---

## Exhaustive grid search

``` python
from sklearn.model_selection import GridSearchCV
```

``` python
param_grid = {
    "gamma": [0.1, 1.0, 10, 100]
}
```

``` python
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, verbose=1)
```

``` python
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 4 candidates, totalling 20 fits

[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  20 out of  20 | elapsed:    0.1s finished
```

Notes:

How do we use
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html" target="_blank">`sklearn.model_selection.GridSearchCV`</a>?

First in the dictionary `param_grid`, we specify the values we wish to
look over.

Next, we build a model of our choosing. Here we are building a SVM
classifier.

Using `GridSearchCV` we first sepecify our model followed by the
hyperparameter values we are checking.

That means `param_grid` for us. We assign `verbose=1` tells
`GridSearchCV` to print some output while it’s working.

---

``` python
param_grid = {
    "gamma": [0.1, 1.0, 10, 100],
    "C": [0.1, 1.0, 10, 100]
}
```

``` python
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv= 5, verbose=1, n_jobs=-1)
```

``` python
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 16 candidates, totalling 80 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.5s
[Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed:    1.5s finished
```

Notes:

The nice thing about this is we can do this for multiple
hyperparameterssimultaneously as well.

So we can search each of the values for `C` and `gamma` while performing
cross-validation\!

In `GridSearchCV` we can specify the number of folds of cross-validation
with the argument `cv`.

Something new we’ve added here is `n_jobs=-1`. This is a little more
complex.

Setting this to -1 helps make this process faster by running
hyperparameter optimization in parallel instead of in a sequence.

Sometimes when we are checking many hyperparameters, values, and with
multiple cross-validation folds, this can take quite a long time.
Setting `n_jobs=-1` helps with that.

---

``` python
pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC())])
```

``` python
param_grid = {
    "svc__gamma": [0.1, 1.0, 10, 100],
    "svc__C": [0.1, 1.0, 10, 100]
}
```

``` python
grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 16 candidates, totalling 80 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed:    0.2s finished
```

Notes:

We can also implement this in with pipelines that we just learned.

After specifying the steps in a pipeline, a user must specify a set of
values for each hyperparameter in `param_grid` like we did before.

Notice that we named our model, `svc` and so we need to call `svc`
followed by 2 underscores and the name of the hyperparameter in
`param_grid` this time.

Now let’s call `GridSearchCV` setting the first argument to the pipeline
name instead of the model name this time.

---

``` python
param_grid = {
    "svc__gamma": [0.1, 1.0, 10, 100],
    "svc__C": [0.1, 1.0, 10, 100]}
    
grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, verbose=1)
```

    for gamma in [0.1, 1.0, 10, 100]:
        for C in [0.1, 1.0, 10, 100]:
            for fold in folds:
                fit in training portion with the given C and gamma
                score on validation portion
            compute average score
    pick hyperparameters with the best score

``` python
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 16 candidates, totalling 80 fits

[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:    0.8s finished
```

Notes:

Looking a bit closer these are the steps being performed with
`GridSearchCV`.

We have 3 loops. That means we are running fit the number of values for
the first hyperparameter multiplied the number of values for the second
hyperparameter multiplied the number of cross-validation folds.

In this case, we can see from the output that 80 execution are done,
just like we calculated 4 x 4 x 5 = 80.

---

<center>

<img src="/module5/cross.gif"  width = "60%" alt="404 image" />

</center>

Notes:

<br>

---

## Now what?

``` python
grid_search.best_params_
```

```out
{'svc__C': 10, 'svc__gamma': 1.0}
```

``` python
grid_search.best_score_
```

```out
0.8208556149732621
```

Notes:

From here, we can extract the best hyperparameter values with
`.best_params_` and its score with `.best_score_`.

We can extract the classifier inside with `.best_estimator_`.

---

``` python
best_model = grid_search.best_estimator_
```

``` python
best_model.fit(X_train, y_train)
```

```out
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()), ('svc', SVC(C=10, gamma=1.0))])
```

``` python
best_model.score(X_test, y_test)
```

```out
0.8333333333333334
```

``` python
grid_search.score(X_test, y_test)
```

```out
0.8333333333333334
```

Notes:

We can extract the classifier inside with `.best_estimator_`.

We can either save it as a new model and fit and score on this new one
*or* we can use the `grid_search` object directly and it will by default
score using the optimal model.

These both give the same results.

---

``` python
best_model.predict(X_test)
```

```out
array(['Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada',
       'Canada'], dtype=object)
```

``` python
grid_search.predict(X_test)
```

```out
array(['Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada',
       'Canada'], dtype=object)
```

Notes:

The same can be done for `.predict()` as well, either using the saved
model, or using the `grid_search` object directly.

---

<br> <br>

### Notice any problems?

  - Required number of models to evaluate grows exponentially with the
    dimensional of the configuration space.
  - Exhaustive search may become infeasible fairly quickly.
  - Example: Suppose we have 5 hyperparameters and 10 different values
    for each hyperparameter
      - That means we’ll be evaluating \(10^5=100,000\) models\! That
        is, we’ll be calling `cross_validate` 100,000 times\!
      - Exhaustive search may become infeasible fairly quickly.

**Enter randomized hyperparameter search\!**

Notes:

This seems pretty nice and obeys the golden rule however the new problem
is the execution time. What if we have many hyperparameters, many values
and we do cross-validation many times? This could take a long time.

<br>

---

``` python
from sklearn.model_selection import RandomizedSearchCV
```

``` python
param_grid = {
    "svc__gamma": [0.1, 1.0, 10, 100],
    "svc__C": [0.1, 1.0, 10, 100]
}
```

``` python
random_search = RandomizedSearchCV(pipe, param_grid, cv=5, verbose=1, n_jobs=-1, n_iter=10)
random_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 10 candidates, totalling 50 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    0.2s finished
```

``` python
random_search.score(X_test, y_test)
```

```out
0.8333333333333334
```

Notes:

Notice that we use the same arguments in `RandomizedSearchCV()` as in
`GridSearchCV()` however with 1 new addition - `n_iter`.

This argument gives us more control and lets us restrict how many
candidates are searched.

Larger `n_iter` will take longer but will do more searching.

Last time when we used exhaustive grid search we had 80 fits (4 x 4 x
5).

This time we see only 50 fits\!

---

# Extra (optional slide)

``` python
import scipy
```

``` python
param_grid = {
    "svc__C": scipy.stats.uniform(0, 100),
    "svc__gamma": scipy.stats.uniform(0, 100)}
```

``` python
random_gs = RandomizedSearchCV(pipe, param_grid, n_jobs=-1, cv=10, return_train_score=True, n_iter=10)
random_gs.fit(X_train, y_train);
```

``` python
random_gs.best_params_
```

```out
{'svc__C': 76.89920365010788, 'svc__gamma': 6.0582080406577195}
```

``` python
random_gs.best_score_
```

```out
0.7867647058823529
```

``` python
random_gs.score(X_test, y_test)
```

```out
0.7380952380952381
```

Notes:

For randomize gridsearch we can search over a range of continuous values
instead of discrete values like in `GridSearchCV()`.

---

## How different do they score?

``` python
grid_search.score(X_test, y_test)
```

```out
0.8333333333333334
```

``` python
random_search.score(X_test, y_test)
```

```out
0.8333333333333334
```

Notes:

---

## Overfitting on validation set

### Overfitting on validation set of parameter learning:

  - During learning, we could search over tons of different decision
    trees.
  - So, we can get “lucky” and find one with a high training score by
    chance.
      - “Overfitting of the training score”.

### Overfitting on validation set of hyper-parameter learning:

  - Here, we might optimize the validation score over 100 values of
    `max_depth`.
  - One of the 100 trees might have a high validation score by chance.

Notes:

Why do we need to evaluate the model on the test set in the end?

Why not just use cross-validation on the whole dataset?

While carrying out hyperparameter optimization, we end up trying over
many possibilities.

If our dataset is small and if our validation set is hit too many times,
we suffer from **optimization bias** or **overfitting the validation
set**.

---

Consider a multiple-choice (a,b,c,d) “test” with 10 questions:

  - If you choose answers randomly, the expected grade is 25% (no bias).
  - If you fill out two tests randomly and pick the best, the expected
    grade is 33%.
      - Optimization bias of \~8%.
  - If you take the best among 10 random tests, the expected grade is
    \~47%.
  - If you take the best among 100, the expected grade is \~62%.
  - If you take the best among 1000, the expected grade is \~73%.
      - You have so many “chances” that you expect to do well.

**But on new questions, the “random choice” accuracy is still 25%.**

  - If we instead used a 100-question test then:
    
      - Expected grade from best over 1 randomly-filled tests is 25%.
      - Expected grade from best over 2 randomly-filled tests is \~27%.
      - Expected grade from best over 10 randomly-filled tests is \~32%.
      - Expected grade from best over 100 randomly-filled tests is
        \~36%.
      - Expected grade from best over 1000 randomly-filled tests is
        \~40%.

  - The optimization bias **grows with the number of things we try**.

  - But, optimization bias **shrinks quickly with the number of
    examples**.
    
      - But it’s still non-zero and growing if you over-use your
        validation set\!

Notes:

---

<br> <br> <br>

<center>

<img src="/module5/optimization_bias.png"  width = "60%" alt="404 image" />

</center>

Notes:

What we can see here is that:

) cross-validation score is too optimistic vs. test score.

The cross-validation scores are too optimistic versus the test score.

The cross-validation score curve is very bumpy, due to the smaller data
set.

The best values of max\_df are different if you look at cv vs. test.

Thus, not only can we not trust the cv scores, we also cannot trust cv’s
ability to choose of the best hyperparameters.

But we don’t have a lot of better alternatives, unfortunately, if we
have a small dataset.

---

# Let’s apply what we learned\!

Notes: <br>

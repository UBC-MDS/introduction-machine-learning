---
type: slides
---

# Automated hyperparameter optimization

Notes: <br>

---

## The problem with hyperparameters

  - We may have a lot of them (e.g. deep learning).
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

<center>

<img src="/module5/cross.gif"  width = "50%" alt="404 image" />

</center>

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
grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, verbose=1)
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 16 candidates, totalling 80 fits

[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:    0.9s finished
```

Notes:
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html" target="_blank">`sklearn.model_selection.GridSearchCV`</a>

How does it work?

After specifying the steps in a pipeline, a user must specify a set of
values for each hyperparameter.

Notice that we named our model, `svc` and so we need to call `svc`
followed by 2 underscores and the name of the hyperparameter.

Now let’s call `GridSearchCV`. There is a lot to look at here:

  - The first argument specifies the pipeline and the steps we wish to
    execute.
  - Next, we specify the hyperparameter values that we wish to search
    over.
  - As we said earlier, the CV in `GridSearchCV` stands for
    cross-validation. We now can set the number of cross-validation
    folds with `cv` and `return_train_score=True` like in
    `cross_validate()`.
  - The next 2 parameters are optional but very useful:
      - `verbose=1` tells `GridSearchCV` to print some output while it’s
        working. This can be useful as this step sometimes takes a long
        time.
      - `n_jobs` is a little more complex. Setting this to -1 helps make
        this process faster by running hyperparameter optimization in
        parallel instead of in a sequence.

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
                fit in training portion with the given C
                score on validation portion
            compute average score
    pick hyperparameters with the best score

``` python
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 16 candidates, totalling 80 fits

[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:    0.9s finished
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

``` python
grid_search.best_estimator_.predict(X_test)
```

```out
array(['Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada',
       'Canada'], dtype=object)
```

or

``` python
grid_search.predict(X_test)
```

```out
array(['Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'Canada', 'Canada', 'Canada', 'Canada', 'Canada', 'USA', 'USA', 'Canada', 'Canada',
       'Canada'], dtype=object)
```

Notes:

From here, we can extract the best hyperparameter values with
`.best_params_` and its score with `.best_score_`.

We can extract the classifier inside with `.best_estimator_`.

---

``` python
grid_search.best_estimator_.score(X_test, y_test)
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

The same can be done for `.score()` as well.

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
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    1.6s finished
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

## Randomized hyperparameter optimization

### Advantages of `RandomizedSearchCV`

  - Faster compared to `GridSearchCV`.
  - Adding parameters that do not influence the performance does not
    affect efficiency.
  - In general, many people recommend using `RandomizedSearchCV` over
    `GridSearchCV`.

<center>

<img src="/module5/randomsearch_bergstra.png"  width = "80%" alt="404 image" />

</center>

<a href="http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf" target="_blank">Source:
Bergstra and Bengio, Random Search for Hyper-Parameter Optimization,
JMLR 2012</a>

Notes:

The yellow on the left shows how our scores are going to change when we
vary the unimportant hyperparameter.

The green on the top shows how our scores are going to change when we
vary the important hyperparameter.

We don’t know in advance which hyperparameters are important for our
problem.

In the left figure, 6 of the 9 searches are useless because they are
only varying the unimportant parameter.

In the right figure, all 9 searches are useful.

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

## Optimization bias

### Optimization bias of parameter learning:

  - During learning, we could search over tons of different decision
    trees.
  - So, we can get “lucky” and find one with a high training score by
    chance.
      - “Overfitting of the training score”.

### Optimization bias of hyper-parameter learning:

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

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Case Study: Pipelines

Notes: <br>

---

``` python
X_train_scaled.head()
```

```out
       longitude  latitude  housing_median_age  households  median_income  rooms_per_household  bedrooms_per_household  population_per_household
6051    0.908140 -0.743917           -0.526078    0.266135      -0.389736            -0.210591               -0.083813                  0.126398
20113  -0.002057  1.083123           -0.923283   -1.253312      -0.198924             4.726412               11.166631                 -0.050132
14289   1.218207 -1.352930            1.380504    0.542873      -0.635239            -0.273606               -0.025391                 -0.099240
13665   1.128188 -0.753286           -0.843842   -0.561467       0.714077             0.122307               -0.280310                  0.010183
14471   1.168196 -1.287344           -0.843842    2.500924      -1.059242            -0.640266               -0.190617                  0.126808
```

``` python
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train);
knn.score(X_train_scaled, y_train).round(3)
```

```out
0.798
```

Notes:

We left off with our scaled data and calculating our training score,
however in the last module we saw that cross-validation is a better way
to get a realistic assessment of the model.

---

### How to carry out cross-validation?

``` python
knn = KNeighborsRegressor()
scores = cross_validate(knn, X_train_scaled, y_train, return_train_score=True)
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.008430    0.172872    0.696373     0.794236
1  0.008192    0.163651    0.684447     0.791467
2  0.007991    0.181362    0.695532     0.789436
3  0.008897    0.172819    0.679478     0.793243
4  0.009800    0.120611    0.680657     0.794820
```

Notes:

Let’s try cross-validation with transformed data.

**Is there a problem here?**

Are we applying `fit_transform` on the train portion and `transform` on
the validation portion in each fold?

Here we might be allowing information from the validation set to
**leak** into the training step.

We need to apply the **SAME** preprocessing steps to train/validation.

With many different transformations and cross-validation, the code gets
unwieldy very quickly.

That makes it likely to make mistakes and “leak” information.

Before we look at the right approach to this, it’s important to look at
the wrong approaches and understand why we cannot perform
cross-validation in such ways.

---

### Bad methodology 1: Scaling the data separately

``` python
scaler = StandardScaler();
scaler.fit(X_train_imp);
X_train_scaled = scaler.transform(X_train_imp)
```

``` python
# Creating a separate object for scaling test data - Not a good idea.
scaler = StandardScaler();
scaler.fit(X_test_imp); # Calling fit on the test data - Yikes! 
```

```out
StandardScaler()
```

``` python
X_test_scaled = scaler.transform(X_test_imp) # Transforming the test data using the scaler fit on test data ... Bad! 
```

``` python
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train);
print("Training score: ", knn.score(X_train_scaled, y_train).round(2))
```

```out
Training score:  0.8
```

``` python
print("Test score: ", knn.score(X_test_scaled, y_test).round(2))
```

```out
Test score:  0.7
```

Notes:

***What is wrong with this approach?***

Although we are keeping our test data separate from our training data,
by scaling the train and test splits separately, this is problematic
since we are using two different `StandardScaler` objects.

This is bad because we want to apply the same transformation on the
training and test splits.

---

### Bad methodology 2: Scaling the data together

``` python
X_train_imp.shape, X_test_imp.shape
```

```out
((18576, 8), (2064, 8))
```

``` python
# Join the train and test sets back together
X_train_imp_df = pd.DataFrame(X_train_imp,columns=X_train.columns, index=X_train.index)
X_test_imp_df = pd.DataFrame(X_test_imp,columns=X_test.columns, index=X_test.index)
XX = pd.concat([X_train_imp_df, X_test_imp_df], axis = 0) ## Don't do it! 
XX.shape 
```

```out
(20640, 8)
```

``` python
scaler = StandardScaler()
scaler.fit(XX);
XX_scaled = scaler.transform(XX) 
XX_train, XX_test = XX_scaled[:18576], XX_scaled[18576:]
```

Notes:

***What is wrong with this second approach?***

Here we are scaling the train and test splits together.

The golden rule says that the test data shouldn’t influence the training
in any way.

With this approach, we are using the information from the test split
when we `fit` the scaler and calculate the mean, as we are passing the
combined `X_train` and `X_test` to it. So, it’s **violation** of the
golden rule.

---

``` python
knn = KNeighborsRegressor()
knn.fit(XX_train, y_train);
print('Train score: ', (knn.score(XX_train, y_train).round(2))) # Misleading score
```

```out
Train score:  0.8
```

``` python
print('Test score: ', (knn.score(XX_test, y_test).round(2))) # Misleading score
```

```out
Test score:  0.71
```

Notes: <br>

---

<br> <br>

### So, what can we do?

We can create a
<a href="ttps://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html" target="_blank">scikit-learn
Pipeline</a>\!

Pipelines allow us to define a “pipeline” of transformers with a final
estimator.

---

## Let’s see it in action

``` python
from sklearn.pipeline import Pipeline
```

``` python
pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor())
])
```

Notes:

We can combine preprocessing and model with a pipeline.

Here is a simple example.

We are passing in a list of steps.

The last step should be a **model/classifier/regressor**.

All the earlier steps should be **transformers**.

---

``` python
pipe.fit(X_train, y_train)
```

```out
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()), ('reg', KNeighborsRegressor())])
```

What’s happening:

``` python
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
scaler = StandardScaler()
scaler.fit(X_train_imp)
X_train_imp_scaled = scaler.transform(X_train_imp)
knn = KNeighborsRegressor()
knn.fit(X_train_imp_scaled)
```

Notes:

Then we fit the `pipe` object and pass in `X_train, y_train`

Notice that we are passing `X_train` and **not** the imputed or scaled
data here.

When we call `fit` the pipeline is carrying out the following steps:

  - Fit `SimpleImputer` on `X_train`.
  - Transform `X_train` using the fit `SimpleImputer` to create
    `X_train_imp`.
  - Fit `StandardScaler` on `X_train_imp`.
  - Transform `X_train_imp` using the fit `StandardScaler` to create
    `X_train_imp_scaled`.
  - Fit the model (`KNeighborsRegressor` in our case) on
    `X_train_imp_scaled`.

---

``` python
pipe.predict(X_train) 
```

```out
array([126500., 117380., 187700., ..., 259500., 308120.,  60860.])
```

``` python
X_train_imp = imputer.transform(X_train)
X_train_imp_scaled = scaler.transform(X_train_imp)
knn.predict(X_train_imp_scaled)
```

Notes:

Take note that when we are passing original data to `predict` the
following steps are carrying out:

  - Transform `X_train` using the fit `SimpleImputer` to create
    `X_train_imp`.
  - Transform `X_train_imp` using the fit `StandardScaler` to create
    `X_train_imp_scaled`.
  - Predict using the fit model (`KNeighborsRegressor` in our case) on
    `X_train_imp_scaled`.

It is not fitting any of the data this time.

---

<center>

<img src="/module5/pipeline.png" width = "70%" alt="404 image" />

</center>

<a href="https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing/#18" target="_blank">Attribution</a>

Notes:

Here is a schematic assuming we have two transformers.

One thing that is awesome with pipelines is that we can’t make the
mistakes we showed earlier.

We call fit on the train split and score on the test split, it’s clean.
We can’t accidentally re-fit the preprocessor on the test data like we
did last time. It automatically makes sure the same transformations are
applied to train and test.

---

``` python
scores_processed = cross_validate(pipe, X_train, y_train, return_train_score=True)
pd.DataFrame(scores_processed)
```

```out
   fit_time  score_time  test_score  train_score
0  0.022225    0.182393    0.693883     0.792395
1  0.021052    0.170285    0.685017     0.789108
2  0.022449    0.178411    0.694409     0.787796
3  0.022472    0.179049    0.677055     0.792444
4  0.020763    0.147359    0.714494     0.823421
```

Notes:

Remember what cross-validation does - it calls fit and score.

Now we’re calling fit on the pipeline, not just the 𝑘-NN regressor.

So, the transformers and the 𝑘-NN model are refit again on each fold.

The pipeline applies the `fit_transform` on the train portion of the
data and only `transform` on the validation portion in each fold.

This is how to avoid the Golden Rule violation\!

---

``` python
pd.DataFrame(scores_processed).mean()
```

```out
fit_time       0.021792
score_time     0.171499
test_score     0.692972
train_score    0.797033
dtype: float64
```

``` python
dummy = DummyRegressor(strategy="median")
scores = cross_validate(dummy, X_train, y_train, return_train_score=True)
pd.DataFrame(scores).mean()
```

```out
fit_time       0.001353
score_time     0.000700
test_score    -0.055115
train_score   -0.054611
dtype: float64
```

Notes:

And we can also see that the preprocessed scores are much better than
our dummy regressor which has negative ones\!

---

# Let’s apply what we learned\!

Notes: <br>

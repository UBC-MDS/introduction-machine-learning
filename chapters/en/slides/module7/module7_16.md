---
type: slides
---

# Regression Measurements

Notes: <br>

---

``` python
housing_df = pd.read_csv("data/housing.csv")
train_df, test_df = train_test_split(housing_df, test_size=0.1, random_state=123)
```

``` python
X_train = train_df.drop(columns=["median_house_value"])
y_train = train_df["median_house_value"]
X_test = test_df.drop(columns=["median_house_value"])
y_test = test_df["median_house_value"]

numeric_features = [ "longitude", "latitude",
                     "housing_median_age",
                     "households", "median_income",
                     "rooms_per_household",
                     "bedrooms_per_household",
                     "population_per_household"]
                     
categorical_features = ["ocean_proximity"]

X_train.head()
```

```out
       longitude  latitude  housing_median_age  households  median_income ocean_proximity  rooms_per_household  bedrooms_per_household  population_per_household
6051     -117.75     34.04                22.0       602.0         3.1250          INLAND             4.897010                1.056478                  4.318937
20113    -119.57     37.94                17.0        20.0         3.4861          INLAND            17.300000                6.500000                  2.550000
14289    -117.13     32.74                46.0       708.0         2.6604      NEAR OCEAN             4.738701                1.084746                  2.057910
13665    -117.31     34.02                18.0       285.0         5.2139          INLAND             5.733333                0.961404                  3.154386
14471    -117.23     32.88                18.0      1458.0         1.8580      NEAR OCEAN             3.817558                1.004801                  4.323045
```

Notes:

This next section involves looking at regression problems and so we are
going to bring back our California housing dataset where we want to
predict the median house value for different locations.

---

``` python
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = make_column_transformer(
(numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features), 
    remainder='passthrough')

pipe = make_pipeline(preprocessor, KNeighborsRegressor())
pipe.fit(X_train, y_train);
```

Notes:

We are going to bring in our previous pipelines and fit our model.

---

``` python
predicted_y = pipe.predict(X_train) 
```

``` python
predicted_y == y_train
```

```out
6051     False
20113    False
14289    False
13665    False
14471    False
         ...  
7763     False
15377    False
17730    False
15725    False
19966    False
Name: median_house_value, Length: 18576, dtype: bool
```

``` python
y_train.values
```

```out
array([113600., 137500., 170100., ..., 286200., 412500.,  59300.])
```

``` python
predicted_y
```

```out
array([111740., 117380., 187700., ..., 271420., 265180.,  60860.])
```

Notes:

We aren’t doing classification anymore, so we can’t just check for
equality.

We need a score that reflects how right/wrong each prediction is or how
close we are to the actual numeric value.

---

## Regression measurements

The scores we are going to discuss are:

  - mean squared error (MSE)
  - R<sup>2</sup>
  - root mean squared error (RMSE)
  - MAPE

If you want to see these in more detail, you can refer to the
<a href="https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics" target="_blank">sklearn
documentation</a>.

Notes:

---

### Mean squared error (MSE)

<center>

<img src="/module7/mse.svg"  width = "20%" alt="404 image" />

</center>

<center>

<img src="/module7/mse-easy.svg"  width = "38%" alt="404 image" />

</center>

``` python
predicted_y
```

```out
array([111740., 117380., 187700., ..., 271420., 265180.,  60860.])
```

``` python
np.mean((y_train - predicted_y)**2)
```

```out
2570054492.048064
```

``` python
np.mean((y_train - y_train)**2)
```

```out
0.0
```

Notes:

Mean Squared Error is a common measure.

We calculate this by calculating the difference between the predicted
and actual value, square it and sum all these values for every example
in the data.

Perfect predictions would have MSE=0.

---

``` python
from sklearn.metrics import mean_squared_error 
```

``` python
mean_squared_error(y_train, predicted_y)
```

```out
2570054492.048064
```

Notes:

We can use `mean_squared_error` from sklearn again instead of
calculating this ourselves.

If we look at MSE here, it’s huge and unreasonable.

Is this score good or bad?

Unlike classification, in regression, our target has units.

In this case, our target column is the median housing value which is in
dollars.

That means that the mean squared error is in dollars<sup>2</sup>.

The score also depends on the scale of the targets.

If we were working in cents instead of dollars, our MSE would be 10,000
X (100<sup>2</sup>) higher\!

---

### R<sup>2</sup> (quick notes)

Key points:

  - The maximum value possible is 1 which means the model has perfect
    predictions.
  - Negative values are very bad: “worse than baseline models such
    as`DummyRegressor`”.

<!-- end list -->

``` python
from sklearn.metrics import r2_score
```

Notes:

This is the score that `sklearn` uses by default when you call
`.score()` so we’ve already seen R<sup>2</sup> in our regression
problems.

You can
<a href="https://en.wikipedia.org/wiki/Coefficient_of_determination" target="_blank">read
about it here</a> but we are going to just give you the quick notes.

Intuition: mean squared error, but flipped where higher values mean a
better measurement.

Normalized so the max is 1.

When you call `fit` it minimizes MSE / maximizes R<sup>2</sup> (or
something like that) by default.

Just like in classification, this isn’t always what you want.

---

``` python
print(mean_squared_error(y_train, predicted_y))
print(mean_squared_error(predicted_y, y_train))
```

``` out
2570054492.048064
2570054492.048064
```

``` python
print(r2_score(y_train, predicted_y))
print(r2_score(predicted_y, y_train))
```

``` out
0.8059396097446094
0.742915970464153
```

Notes:

We can reverse MSE but not R<sup>2</sup> (optional).

---

### Root mean squared error (RMSE)

<center>

<img src="/module7/rmse-simp.svg"  width = "20%" alt="404 image" />

</center>

<center>

<img src="/module7/mse-easy.svg"  width = "38%" alt="404 image" />

</center>

``` python
mean_squared_error(y_train, predicted_y)
```

```out
2570054492.048064
```

``` python
np.sqrt(mean_squared_error(y_train, predicted_y))
```

```out
50695.704867849156
```

Notes:

The MSE we had before was in dollars<sup>2</sup>.

A more relatable metric would be the root mean squared error, or RMSE.

This now has the units in dollars. Instead of 250 million dollars
squared our error measurement is around $50,000.

---

<img src="/module6/module7_16/unnamed-chunk-19-1.png" width="75%" style="display: block; margin: auto;" />

Notes:

When we plot our predictions versus the examples’ actual value, we can
see cases where our prediction is way off.

Under the line means we’re under-prediction, over the line means we’re
over-predicting.

Question: Is an error of $30,000 acceptable?

  - For a house worth $600k, it seems reasonable\! That’s a 5% error.
  - For a house worth $60k, that is terrible. It’s a 50% error.

---

### MAPE - Mean Absolute Percent Error (MAPE)

``` python
percent_errors = (predicted_y - y_train)/y_train * 100.
percent_errors.head()
```

```out
6051     -1.637324
20113   -14.632727
14289    10.346855
13665     6.713070
14471   -10.965854
Name: median_house_value, dtype: float64
```

``` python
np.abs(percent_errors).head()
```

```out
6051      1.637324
20113    14.632727
14289    10.346855
13665     6.713070
14471    10.965854
Name: median_house_value, dtype: float64
```

``` python
100.*np.mean(np.abs((predicted_y - y_train)/y_train))
```

```out
18.192997502985218
```

Notes:

So, finding the percentage error may be handy. Can we compute something
like that?

We can calculate a percentage error for each example. Now the errors are
both positive (predict too high) and negative (predict too low).

We can look at the absolute percent error which now shows us how far off
we were independent of direction.

Like MSE, we can take the average over all the examples. This is called
**Mean Absolute Percent Error (MAPE)**.

Ok, this is quite interpretable. We can see that on average, we have
around 18% error in our predicted median housing valuation.

---

# Let’s apply what we learned\!

Notes: <br>

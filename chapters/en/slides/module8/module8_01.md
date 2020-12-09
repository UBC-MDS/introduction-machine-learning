---
type: slides
---

# Introducing linear regression

Notes: <br>

---

## Linear Regression

``` python
train_df.head()
```

```out
      length     weight
73  1.489130  10.507995
53  1.073233   7.658047
80  1.622709   9.748797
49  0.984653   9.731572
23  0.484937   3.016555
```

Notes:

We’ve seen many regression models such as `DecisionTreeRegressor` and
`KNeighborsRegressor` but now we have a new one that we are going to
explore called  
**linear regression**.

Linear regression is one of the most basic and popular ML/statistical
techniques.

Let’s bring back the hypothetical snake data that we saw in module 4.

---

## Ridge

``` python
from sklearn.linear_model import LinearRegression
LinearRegression();
```

``` python
from sklearn.linear_model import Ridge
```

``` python
rm = Ridge()
rm.fit(X_train, y_train);
```

``` python
rm.predict(X_train)[:5]
```

```out
array([10.09739051,  7.90823334, 10.80050927,  7.44197529,  4.81162144])
```

``` python
rm.score(X_train, y_train)
```

```out
0.8125029624787177
```

Notes:

We can import the `LinearRegression` model like we have for all the
previous models we’ve used except we are going to instead focus on its
close cousin `Ridge`.

`Ridge` is more flexible than `LinearRegression` and we will explain why
shortly.

When we import `Ridge`, you’ll notice that we are importing from the
`linear_model` Sklearn library.

`Ridge`, has the same fit-predict paradigm as the other models we have
seen.

That means we can `fit` on the training set and `predict` a numeric
prediction.

We see that `predict` returns the predicted housing prices for our
examples.

---

## *alpha*

``` python
rm2 = Ridge(alpha=10000)
rm2.fit(X_train, y_train);
```

``` python
rm2.score(X_train, y_train)
```

```out
0.004541128724857568
```

Notes:

Ridge has hyperparameters just like the rest of the models we learned.

The `alpha` hyperparameter is what makes it more flexible than using
`LinearRegression`.

Remember the fundamental trade-off we spoke about in module 3?

**“As model complexity ↑, training score ↑ and training score–
validation score tend to ↑”**

Well, `alpha` controls this fundamental trade-off\!

---

``` python
scores_dict ={
"alpha" :10.0**np.arange(-2,6,1),
"train_scores" : list(),
"cv_scores" : list(),
}
for alpha in scores_dict['alpha']:
    ridge_model = Ridge(alpha=alpha)
    results = cross_validate(ridge_model, X_train, y_train, return_train_score=True)
    scores_dict['train_scores'].append(results["train_score"].mean())
    scores_dict['cv_scores'].append(results["test_score"].mean())
```

``` python
pd.DataFrame(scores_dict)
```

```out
       alpha  train_scores  cv_scores
0       0.01      0.812961   0.799169
1       0.10      0.812945   0.799199
2       1.00      0.811461   0.798103
3      10.00      0.735071   0.721655
4     100.00      0.270059   0.244916
5    1000.00      0.035217   0.003744
6   10000.00      0.003629  -0.028689
7  100000.00      0.000364  -0.032041
```

Notes:

As we increase `alpha`, we are decreasing our model complexity which
means our training score is lower and we are more likely to underfit.

If we decrease `alpha`, our model complexity is increasing and
consequentially our training score is increasing. Our chances of
overfitting are going up.

---

# Visualizing linear regression

<img src="/module8/module8_01/unnamed-chunk-14-1.png" width="80%" style="display: block; margin: auto;" />

Notes:

In our data, we only have 1 feature `length` which helps predict our
target feature `weight`.

We can use a 2D graph to plot this and our ridge regression corresponds
to a line.

In this plot, the blue markers are the examples and our orange line is
our Ridge regression line.

If we had an additional feature, let’s say, `width`, we now would have 2
features and 1 target so our ridge regression would correspond to a plan
in a 3-dimensional space.

As we increase our features beyond 3 it becomes harder to visualize.

---

# Let’s apply what we learned\!

Notes: <br>

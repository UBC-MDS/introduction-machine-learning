---
type: slides
---

# Decision Tree Regressor

Notes: <br>

---

``` python
regression_df = pd.read_csv("data/quiz2-grade-toy-regression.csv")
regression_df
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1  quiz2
0              1                 1    92    93    84    91     92     90
1              1                 0    94    90    80    83     91     84
2              0                 0    78    85    83    80     80     82
3              0                 1    91    94    92    91     89     92
4              0                 1    77    83    90    92     85     90
5              1                 0    70    73    68    74     71     75
6              1                 0    80    88    89    88     91     91
```

Notes:

We can also use the decision tree algorithm for regression problems.

Instead of using gini (which we briefly mentioned this in previous
slides), we can use
<a href="https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation" target="_blank">some
other criteria</a> for splitting. (A common one is mean squared error
(MSE) which we will discuss shortly)

`scikit-learn` supports regression using decision trees with
`DecisionTreeRegressor()` and the `.fit()` and `.predict()` paradigm
that is similar to classification.

Just like when we talked about the baseline Dummy regressor, `.score()`
for regression returns somethings called an
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score" target="_blank">
𝑅2 </a>.

The maximum 𝑅2 is 1 for perfect predictions. It can be negative which is
very bad (worse than DummyRegressor).

---

``` python
X = regression_df.drop(["quiz2"], axis=1)
X.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
1              1                 0    94    90    80    83     91
2              0                 0    78    85    83    80     80
3              0                 1    91    94    92    91     89
4              0                 1    77    83    90    92     85
```

``` python
y = regression_df["quiz2"]
y.head()
```

```out
0    90
1    84
2    82
3    92
4    90
Name: quiz2, dtype: int64
```

Notes:

**Decision Tree Regressor** is built using `DecisionTreeRegressor()` and
a similar `.fit()` and `.predict()` paradigms as classification.

We split our data just like we did before into our feature table and our
target.

---

``` python
from sklearn.tree import DecisionTreeRegressor
depth = 4
reg_model = DecisionTreeRegressor(max_depth=depth)
reg_model.fit(X, y)
```

Notes:

Instead of importing `DecisionTreeClassifier`, we import
`DecisionTreeRegressor`.

We follow the same steps before and can even set hyperparameters as we
did in classification.

Here when we build our constructor, we are specifying a `max_depth` of
4.

---

``` python
display_tree(X.columns, reg_model, "/module2/module2_16a")
```

```out
<graphviz.files.Source object at 0x127331910>

/usr/local/lib/python3.8/site-packages/sklearn/tree/_classes.py:1254: FutureWarning: the classes_ attribute is to be deprecated from version 0.22 and will be removed in 0.24.
  warnings.warn(msg, FutureWarning)
```

<center>

<img src="/module2/module2_16a.png"  width = "80%" alt="404 image" />

</center>

Notes:

---

``` python
reg_model.predict(X.loc[[0]])
```

```out
array([90.])
```

``` python
predicted_grades = reg_model.predict(X)
regression_df['predicted_quiz2'] = predicted_grades
print("R^2 score on the training data:" + str(round(reg_model.score(X,y), 3)))
```

```out
R^2 score on the training data:1.0
```

``` python
regression_df.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1  quiz2  predicted_quiz2
0              1                 1    92    93    84    91     92     90             90.0
1              1                 0    94    90    80    83     91     84             84.0
2              0                 0    78    85    83    80     80     82             82.0
3              0                 1    91    94    92    91     89     92             92.0
4              0                 1    77    83    90    92     85     90             90.0
```

Notes:

This time when we predict a single observation, instead of getting a
prediction that is a category or class, we are getting a numerical
value.

Here we see that the model predicts a grade of 90 for the first
observation.

Let’s see how well this model does predicting the entire data.

Since the 𝑅2 value is 1.0, the model perfectly predicts the outcome of
every observation.

This is quite different from what we were getting with a Dummy
Classifier which had an 𝑅2 value of 0.

---

# Let’s apply what we learned\!

Notes: <br>

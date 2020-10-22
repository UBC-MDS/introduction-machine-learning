---
type: slides
---

# Dummy Regression

Notes: <br>

---

### Building a baseline regression model

<br> <br> <br> Baseline model: **Averge target value**: always predicts
the mean of the training set.

Notes:

Let’s build a ***baseline*** simple machine learning algorithm based on
simple rules of thumb.

For a regression problem, we are going to build a baseline model that
always predicts the mean target value in the training set.

---

## Data

``` python
classification_df = pd.read_csv("data/quiz2-grade-toy-regression.csv")
classification_df.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1  quiz2
0              1                 1    92    93    84    91     92     90
1              1                 0    94    90    80    83     91     84
2              0                 0    78    85    83    80     80     82
3              0                 1    91    94    92    91     89     92
4              0                 1    77    83    90    92     85     90
```

Notes:

Let’s bring in our regression problem data now.

For this example, we are going to be working with the quiz2 regression
data that we have seen previously.

---

## 1\. Create 𝑋 and 𝑦

𝑋 → Feature vectors <br> 𝑦 → Target

``` python
X = classification_df.drop(columns=["quiz2"])
y = classification_df["quiz2"]
```

Notes:

Just like before, we separate our data into the feature table and the
target, also known as 𝑋 and 𝑦.

<br>

---

## 2\. Create a regressor object

  - `import` the appropriate regressor, in this case, `DummyRegressor`.
  - Create an object of the regressor.

<!-- end list -->

``` python
from sklearn.dummy import DummyRegressor

dummy_reg = DummyRegressor(strategy="mean")
```

Notes:

This time instead of importing `DummyClassifier()`, we import
`DummyRegessor` which will be used to create our baseline regression
model.

We specify in the `strategy` argument `mean`. With this strategy, the
model will predict the mean target value in the training data.

Here we are naming our model `dummy_reg`.

---

## 3\. Fit the regressor

``` python
dummy_reg.fit(X, y)
```

Notes:

Once we have created and named our regressor, we give it data to train
on.

---

## 4\. Predict the target of given examples

We can predict the mean of examples by calling `predict` on the
classifier object.

``` python
single_obs = X.loc[[2]]
single_obs
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
2              0                 0    78    85    83    80     80
```

``` python
dummy_reg.predict(single_obs)
```

```out
array([86.28571429])
```

Notes:

Since our model has been trained on existing data, we can predict the
targets.

For observation 2, the model predicts a value of `86.29`

---

``` python
X
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
1              1                 0    94    90    80    83     91
2              0                 0    78    85    83    80     80
3              0                 1    91    94    92    91     89
4              0                 1    77    83    90    92     85
5              1                 0    70    73    68    74     71
6              1                 0    80    88    89    88     91
```

``` python
dummy_reg.predict(X)
```

```out
array([86.28571429, 86.28571429, 86.28571429, 86.28571429, 86.28571429, 86.28571429, 86.28571429])
```

Notes:

In fact, the model predicts the same value for all the observations.

---

## 5\. Scoring your model

In the regression setting, `.score()` gives the \(R^2\) of the model,
i.e. the coefficient of determination of the prediction.

``` python
print("The accuracy of the model on the training data: %0.3f" %(dummy_reg.score(X, y)))
```

```out
The accuracy of the model on the training data: 0.000
```

Notes:

The best possible score for any model is 1.0 and it can be a negative
(because the model can be arbitrarily worse).

We will talk about this value further in the course, but for now, all
you need to know is that for dummy regressors, the output of `.score()`
when using a dummy regressor with a `strategy` argument value of `mean`,
will be 0.0.

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Overfitting and underfitting

Notes: <br>

---

We’re going to think about 3 types of errors:

  - **score\_train**: is our training score (or mean train score from
    cross-validation).

<br>

  - **score\_valid** is our validation score (or mean validation score
    from cross-validation).

<br>

  - **score\_test** is our test score.

Notes:

We’ve talked about the different types of splits but we’ve only briefly
discussed scores and the different types of scores that we receive when
building models.

We saw in cross-validation that there was train and validation scores
and what happens if they did not align with each other.

How do we diagnose the problem?

We’re going to think about 3 types of scores:

  - **score\_train**: is our training score (or mean train score from
    cross-validation).
  - **score\_valid** is our validation score (or mean validation score
    from cross-validation).
  - **score\_test** is our test score

---

``` python
df = pd.read_csv("data/canada_usa_cities.csv")
X = df.drop(columns=["country"])
y = df["country"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
```

Notes:

Let’s bring back our Canadian and United States cities’ data to help
explain the concepts of overfitting and underfitting.

---

## Overfitting

``` python
model = DecisionTreeClassifier()
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
print("Train score: " + str(round(scores["train_score"].mean(), 2)))
print("Validation score: " + str(round(scores["test_score"].mean(), 2)))
```

``` out
Train score: 1.0
Validation score: 0.81
```

<img src="/module3/module3_13/unnamed-chunk-3-1.png" width="70%" />

Notes:

Using a decision tree with no specified max\_depth, we can explain the
phenomenon is called ***overfitting***.

Overfitting is when our model fits the training data well and therefore
the training score is high, however, the model does not generalize to
the validation set as well and the validation error is much higher.

The Train score is high but the validation score is much lower.

The gap between the train and validation scores is bigger.

A standard overfitting scenario would be:
**Score\_train\>\>Score\_valid**

If **Score\_train** is high, then we are in an overfitting scenario. It
is fairly common to have at least a bit of this.

---

## Underfitting

``` python
model = DecisionTreeClassifier(max_depth=1)

scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
print("Train score: " + str(round(scores["train_score"].mean(), 2)))
print("Validation score: " + str(round(scores["test_score"].mean(), 2)))
```

``` out
Train score: 0.83
Validation score: 0.81
```

<img src="/module3/module3_13/unnamed-chunk-4-1.png" width="70%" />

Notes:

Using a decision tree with a max\_depth of 1, we can explain the
phenomenon is called ***underfitting***.

Underfitting is when our model is too simple (`DecisionTreeClassifier`
with max\_depth=1 or `DummyClassifier`).

The model doesn’t capture the patterns in the training data and the
training score is not that high.

The model doesn’t fit the data well and hence
**Score\_valid≲Score\_train**.

Both train and validation scores are bad and the gap between train and
validation scores is lower.

**\<Score\_valid≲Score\_train**

---

<center>

<img src="/module3/over_under.png"  width = "80%" alt="404 image" />

</center>

Standard question to ask ourselves: ***Which of these scenarios am I
in?***

### How can we figure this out?

Score\_train and Score\_valid.

  - If they are very far apart → more likely **overfitting**.
      - Try decreasing model complexity.
  - If they are very close together → more likely **underfitting**.
      - Try increasing model complexity.

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Overfitting and underfitting

Notes: <br>

---

We’re going to think about 4 types of errors:

  - **𝐸\_train**: is our training error (or mean train error from
    cross-validation).
  - **𝐸\_valid** is our validation error (or mean validation error from
    cross-validation).
  - **𝐸\_test** is our test error.
  - **𝐸\_best** is the best possible error we could get for a given
    problem.

Question: Why is 𝐸\_best\>0?

Notes:

We’ve talked about the different types of splits but we’ve only briefly
discussed error and the different types of error that we receive when
building models.

We saw in cross-validation that there was train and validation error and
image if they did not align with each other.

How do we diagnose the problem?

We’re going to think about 4 types of errors:

  - 𝐸\_train is our training error (or mean train error from
    cross-validation).
  - 𝐸\_valid is our validation error (or mean validation error from
    cross-validation).
  - 𝐸\_test is our test error.
  - 𝐸\_best is the best possible error we could get for a given problem.

---

``` python
df = pd.read_csv("data/canada_usa_cities.csv")
X = df.drop(["country"], axis=1)
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
print("Train error:   %0.3f" % (1 - np.mean(scores["train_score"])))
print("Validation error:   %0.3f" % (1 - np.mean(scores["test_score"])))
```

``` out
Train error:   0.000
Validation error:   0.191
```

<img src="/module3/module3_13/unnamed-chunk-3-1.png" width="78%" />

Notes:

Using a decision tree with no specified max\_depth, we can explain the
phenomenon is called ***overfitting***.

Overfitting is when our model fits the training data well and therefore
the training error is low, however, the model does not generalize to the
validation set as well and the validation error is much higher.

The Train error is low but the validation error is much higher.

The gap between the train and validation error is bigger.

A standard overfitting scenario would be:
**𝐸\_train\<𝐸\_best\<𝐸\_valid**

If 𝐸\_train is low, then we are in an overfitting scenario. It is fairly
common to have at least a bit of this

𝐸\_valid cannot be smaller than 𝐸\_best basically by definition. In
reality, we won’t have them equal.

---

## Underfitting

``` python
model = DecisionTreeClassifier(max_depth=1)

scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
print("Train error: " + str(round(1 - np.mean(scores["train_score"]),2)))
print("Validation error: "  + str(round(1 - np.mean(scores["test_score"]),2)))
```

``` out
Train error: 0.17
Validation error: 0.19
```

<img src="/module3/module3_13/unnamed-chunk-4-1.png" width="78%" />

Notes:

Using a decision tree with a max\_depth of 1, we can explain the
phenomenon is called ***underfitting***.

Underfitting is when our model is too simple (`DecisionTreeClassifier`
with max\_depth=1 or `DummyClassifier`).

The model doesn’t capture the patterns in the training data and the
training error is not that low.

The model doesn’t fit the data well and hence 𝐸\_train≲𝐸\_valid.

Both train and validation errors are bad and the gap between train and
validation error is lower.

**𝐸\_best\<𝐸\_train≲𝐸\_valid**

---

<center>

<img src="/module3/over_under.png"  width = "80%" alt="404 image" />

</center>

Standard question to ask ourselves: ***Which of these scenarios am I
in?***

### How can we figure this out?

We can’t see 𝐸\_best but we can see 𝐸\_train and 𝐸\_test.

  - If they are very far apart → more likely **overfitting**.
      - Try decreasing model complexity.
  - If they are very close together → more likely **underfitting**.
      - Try increasing model complexity.

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

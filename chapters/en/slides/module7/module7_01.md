---
type: slides
---

# Introducing evaluation metrics

Notes: <br>

---

``` python
cc_df = pd.read_csv('data/creditcard.csv', encoding='latin-1')
train_df, test_df = train_test_split(cc_df, test_size=0.3, random_state=111)
```

``` python
train_df.head()
```

```out
            Time        V1        V2        V3        V4        V5  ...       V25       V26       V27       V28  Amount  Class
64454    51150.0 -3.538816  3.481893 -1.827130 -0.573050  2.644106  ...  0.729143 -0.547993 -0.023636 -0.454966    1.00      0
37906    39163.0 -0.363913  0.853399  1.648195  1.118934  0.100882  ... -0.944092 -0.558564 -0.186814 -0.257103   18.49      0
79378    57994.0  1.193021 -0.136714  0.622612  0.780864 -0.823511  ...  0.254443  0.290002 -0.036764  0.015039   23.74      0
245686  152859.0  1.604032 -0.808208 -1.594982  0.200475  0.502985  ...  0.037785  0.061206  0.005387 -0.057296  156.52      0
60943    49575.0 -2.669614 -2.734385  0.662450 -0.059077  3.346850  ... -1.174590  0.573818  0.388023  0.161782   57.50      0

[5 rows x 31 columns]
```

``` python
train_df.shape
```

```out
(199364, 31)
```

Notes:

Up until this point, we have been scoring our models the same way every
time.

We’ve been using the percentage of correctly predicted examples for
classification problems and the R<sup>2</sup> metric for regression
problems.

To help explain why this isn’t the most beneficial option, we are
bringing in a new dataset.

Let’s classify fraudulent and non-fraudulent transactions using a
<a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank">credit
card fraud detection data set</a>.

We can see this is a large dataset with 199364 examples and 31 features
in our training set.

---

``` python
train_df.describe(include="all", percentiles = [])
```

```out
                Time             V1             V2             V3             V4             V5  ...            V25            V26            V27            V28         Amount          Class
count  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  ...  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000
mean    94888.815669       0.000492      -0.000726       0.000927       0.000630       0.000036  ...       0.000235       0.000312      -0.000366       0.000227      88.164679       0.001700
std     47491.435489       1.959870       1.645519       1.505335       1.413958       1.361718  ...       0.520857       0.481960       0.401541       0.333139     238.925768       0.041201
min         0.000000     -56.407510     -72.715728     -31.813586      -5.683171     -42.147898  ...     -10.295397      -2.241620     -22.565679     -11.710896       0.000000       0.000000
50%     84772.500000       0.018854       0.065463       0.179080      -0.019531      -0.056703  ...       0.016587      -0.052790       0.001239       0.011234      22.000000       0.000000
max    172792.000000       2.451888      22.057729       9.382558      16.491217      34.801666  ...       6.070850       3.517346      12.152401      33.847808   11898.090000       1.000000

[6 rows x 31 columns]
```

Notes:

We see that the columns are all scaled and numerical.

You don’t need to worry about this now. The original columns have been
transformed already for confidentiality and our benefit so now there are
no categorical features.

---

``` python
X_train_big, y_train_big = train_df.drop(columns=["Class"]), train_df["Class"]
X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]
```

``` python
X_train, X_valid, y_train, y_valid = train_test_split(X_train_big, 
                                                      y_train_big, 
                                                      test_size=0.3, 
                                                      random_state=123)
```

Notes:

Let’s separate `X` and `y` for train and test splits.

It’s easier to demonstrate evaluation metrics using an explicit
validation set instead of using cross-validation.

Our data is large enough so it shouldn’t be a problem.

---

### Baseline

``` python
dummy = DummyClassifier(strategy="most_frequent")
pd.DataFrame(cross_validate(dummy, X_train, y_train, return_train_score=True)).mean()
```

```out
fit_time       0.013017
score_time     0.001685
test_score     0.998302
train_score    0.998302
dtype: float64
```

``` python
train_df["Class"].value_counts(normalize=True)
```

```out
0    0.9983
1    0.0017
Name: Class, dtype: float64
```

Notes:

We build a simple `DummyClassifier` model as our baseline but what is
going on? We are getting 99.8% accuracy\!

Should we be happy with this accuracy and deploy this `DummyClassifier`
model for fraud detection?

If we look at the distribution of fraudulent labels to non-fraudulent
labels, we can see there is an imbalance in the classes.

There are MANY non-fraud transactions and only a tiny handful of fraud
transactions.

So, what would be a good accuracy here? 99.9%? 99.99%?

The “fraud” class is the class that we want to spot.

This module will tackle this issue.

---

``` python
pipe = make_pipeline(
       (StandardScaler()),
       (DecisionTreeClassifier(random_state=123))
)
```

``` python
pd.DataFrame(cross_validate(pipe, X_train, y_train, return_train_score=True)).mean()
```

```out
fit_time       6.130222
score_time     0.009888
test_score     0.999119
train_score    1.000000
dtype: float64
```

Notes:

We can make a model better than the dummy classifier now and we get
similar results.

This seems slightly better than `DummyClassifier`, but can it really
identify fraudulent transactions?

This model will cover new tools on how to measure this.

---

### What is “positive” and “negative”?

``` python
train_df["Class"].value_counts(normalize=True)
```

```out
0    0.9983
1    0.0017
Name: Class, dtype: float64
```

There are two kinds of binary classification problems:

  - Distinguishing between two classes
  - Spotting a class (fraud transaction, spam, disease)

Notes:

In the case of spotting problems, the thing that we are interested in
spotting is considered “positive”.

In our example, we want to spot fraudulent transactions and so they are
“positive”.

---

# Confusion Matrix

``` python
pipe.fit(X_train, y_train);
```

``` python
from sklearn.metrics import  plot_confusion_matrix
```

``` python
plot_confusion_matrix(pipe, X_valid, y_valid, display_labels=["Non fraud", "Fraud"], values_format="d", cmap="Blues");
```

<img src="/module7/module7_01/unnamed-chunk-16-1.png" width="50%" />

Notes:

A **confusion matrix** is a table that visualizes the performance of an
algorithm. It shows the possible labels and how many of each label the
model predicts correctly and incorrectly.

Once we fit on our training portion, we can use the
`plot_confusion_matrix` function from sklearn.

In this case, we are looking at the validation portion only.

This results in a 2 by 2 matrix with the labels `Non fraud` and `Fraud`
on each axis.

#### Careful:

Scikit-learn’s convention is to have the true label as the rows and the
predicted label as the columns.

Others do it the other way around, e.g., the confusion matrix
<a href=" https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank">Wikipedia
article</a> .

---

<img src="/module7/module7_01/unnamed-chunk-17-1.png" width="60%" style="display: block; margin: auto;" />

| X                | predict negative    | predict positive    |
| ---------------- | ------------------- | ------------------- |
| negative example | True negative (TN)  | False positive (FP) |
| positive example | False negative (FN) | True positive (TP)  |

Notes:

Remember the Fraud is considered “positive” in this case and non-fraud
is considered “negative”.

Here the 4 quadrants for this problem are explained below. These
positions will change depending on what values we deem as the positive
label.

  - **True negative (TN)**: Examples that are negatively labeled that
    the model correctly predicts. This is in the top left quadrant.
  - **False positive (FP)**: Examples that are negatively labeled that
    the model incorrectly predicts as positive. This is in the top right
    quadrant.
  - **False negative (FN)**: Examples that are positively labeled that
    the model incorrectly predicts as negative. This is in the bottom
    left quadrant.
  - **True positive (TP)**: Examples that are positively labeled that
    the model correctly predicted as positive This is in the bottom
    right quadrant.

---

``` python
from sklearn.metrics import confusion_matrix
```

``` python
predictions = pipe.predict(X_valid)
confusion_matrix(y_valid, predictions)
```

```out
array([[59674,    34],
       [   26,    76]])
```

Notes:

If you want something more numeric and simpler you can obtain a NumPy
array by importing `confusion_matrix` from the sklearn library.

Here we get the predictions of the model first with `.predict()` and
compare it with `y_valid` in the function `confusion_matrix()`.

---

# Let’s apply what we learned\!

Notes: <br>

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
            Time        V1        V2        V3        V4        V5        V6        V7        V8        V9       V10       V11       V12       V13       V14       V15       V16       V17       V18       V19       V20       V21       V22       V23       V24       V25       V26       V27       V28  Amount  Class
64454    51150.0 -3.538816  3.481893 -1.827130 -0.573050  2.644106 -0.340988  2.102135 -2.939006  2.578654  3.155261  0.469895 -1.170292  0.342072 -5.854435  1.090875  0.530014  1.516968  1.014804 -1.509991  1.345904  0.530978 -0.860677 -0.201810 -1.719747  0.729143 -0.547993 -0.023636 -0.454966    1.00      0
37906    39163.0 -0.363913  0.853399  1.648195  1.118934  0.100882  0.423852  0.472790 -0.972440  0.033833  0.629036  1.257913 -0.161244 -2.001477  0.260308  0.189953 -0.844019  0.173619 -0.021524  0.810267 -0.192932  0.687055 -0.094586  0.121531  0.146830 -0.944092 -0.558564 -0.186814 -0.257103   18.49      0
79378    57994.0  1.193021 -0.136714  0.622612  0.780864 -0.823511 -0.706444 -0.206073 -0.016918  0.781531 -0.185059 -0.904320 -0.341810 -1.589398  0.193059  0.043635 -0.104751  0.101689 -0.555039  0.258815 -0.178761 -0.310405 -0.842028  0.085477  0.366005  0.254443  0.290002 -0.036764  0.015039   23.74      0
245686  152859.0  1.604032 -0.808208 -1.594982  0.200475  0.502985  0.832370 -0.034071  0.234040  0.550616 -0.051983  1.253511  0.986015 -0.417782  0.737555  0.711161 -0.916659  0.130676 -0.776092 -1.009429 -0.040448  0.519029  1.429217 -0.139322 -1.293663  0.037785  0.061206  0.005387 -0.057296  156.52      0
60943    49575.0 -2.669614 -2.734385  0.662450 -0.059077  3.346850 -2.549682 -1.430571 -0.118450  0.469383 -0.185599 -1.309417  0.704059  1.245003 -0.334923 -0.333972  0.342296 -0.717449 -0.590856  0.157993 -0.430295 -0.228329 -0.370643 -0.211544 -0.300837 -1.174590  0.573818  0.388023  0.161782   57.50      0
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
classification problems and the R^2 metric for regression problems.

To help explain why this isn’t the most beneficial option, we are
bringing in a new dataset.

Let’s classify fraudulent and non-fraudulent transactions using a
<a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank">credit
card fraud detection data set</a>.

We can see this is a large dataset with 242085 examples and 31 features
in our training set.

---

``` python
train_df.head()
```

```out
            Time        V1        V2        V3        V4        V5        V6        V7        V8        V9       V10       V11       V12       V13       V14       V15       V16       V17       V18       V19       V20       V21       V22       V23       V24       V25       V26       V27       V28  Amount  Class
64454    51150.0 -3.538816  3.481893 -1.827130 -0.573050  2.644106 -0.340988  2.102135 -2.939006  2.578654  3.155261  0.469895 -1.170292  0.342072 -5.854435  1.090875  0.530014  1.516968  1.014804 -1.509991  1.345904  0.530978 -0.860677 -0.201810 -1.719747  0.729143 -0.547993 -0.023636 -0.454966    1.00      0
37906    39163.0 -0.363913  0.853399  1.648195  1.118934  0.100882  0.423852  0.472790 -0.972440  0.033833  0.629036  1.257913 -0.161244 -2.001477  0.260308  0.189953 -0.844019  0.173619 -0.021524  0.810267 -0.192932  0.687055 -0.094586  0.121531  0.146830 -0.944092 -0.558564 -0.186814 -0.257103   18.49      0
79378    57994.0  1.193021 -0.136714  0.622612  0.780864 -0.823511 -0.706444 -0.206073 -0.016918  0.781531 -0.185059 -0.904320 -0.341810 -1.589398  0.193059  0.043635 -0.104751  0.101689 -0.555039  0.258815 -0.178761 -0.310405 -0.842028  0.085477  0.366005  0.254443  0.290002 -0.036764  0.015039   23.74      0
245686  152859.0  1.604032 -0.808208 -1.594982  0.200475  0.502985  0.832370 -0.034071  0.234040  0.550616 -0.051983  1.253511  0.986015 -0.417782  0.737555  0.711161 -0.916659  0.130676 -0.776092 -1.009429 -0.040448  0.519029  1.429217 -0.139322 -1.293663  0.037785  0.061206  0.005387 -0.057296  156.52      0
60943    49575.0 -2.669614 -2.734385  0.662450 -0.059077  3.346850 -2.549682 -1.430571 -0.118450  0.469383 -0.185599 -1.309417  0.704059  1.245003 -0.334923 -0.333972  0.342296 -0.717449 -0.590856  0.157993 -0.430295 -0.228329 -0.370643 -0.211544 -0.300837 -1.174590  0.573818  0.388023  0.161782   57.50      0
```

``` python
train_df.describe(include="all", percentiles = [])
```

```out
                Time             V1             V2             V3             V4             V5             V6             V7             V8             V9            V10            V11            V12            V13            V14            V15            V16            V17            V18            V19            V20            V21  \
count  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000   
mean    94888.815669       0.000492      -0.000726       0.000927       0.000630       0.000036       0.000011      -0.001286      -0.002889      -0.000891       0.000776      -0.001212       0.002526      -0.000710      -0.000887      -0.000835       0.001635       0.001919       0.000566      -0.001132      -0.001995       0.001205   
std     47491.435489       1.959870       1.645519       1.505335       1.413958       1.361718       1.327188       1.210001       1.214852       1.096927       1.083794       1.019130       0.993421       0.996563       0.956001       0.915559       0.872982       0.841571       0.835615       0.814356       0.766957       0.748510   
min         0.000000     -56.407510     -72.715728     -31.813586      -5.683171     -42.147898     -26.160506     -43.557242     -73.216718     -13.320155     -24.588262      -4.797473     -18.683715      -5.791881     -19.214325      -4.498945     -14.129855     -25.162799      -9.287832      -4.932733     -28.009635     -34.830382   
50%     84772.500000       0.018854       0.065463       0.179080      -0.019531      -0.056703      -0.275290       0.040497       0.022039      -0.052607      -0.092421      -0.033516       0.141313      -0.014469       0.049526       0.048555       0.067309      -0.065647      -0.003900       0.004125      -0.062702      -0.029146   
max    172792.000000       2.451888      22.057729       9.382558      16.491217      34.801666      23.917837      44.054461      19.587773      15.594995      23.745136      12.018913       7.848392       4.569009      10.526766       5.825654       8.289890       9.253526       4.295648       5.591971      24.133894      27.202839   

                 V22            V23            V24            V25            V26            V27            V28         Amount          Class  
count  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  199364.000000  
mean        0.000155      -0.000198       0.000113       0.000235       0.000312      -0.000366       0.000227      88.164679       0.001700  
std         0.726634       0.628139       0.605060       0.520857       0.481960       0.401541       0.333139     238.925768       0.041201  
min        -8.887017     -44.807735      -2.824849     -10.295397      -2.241620     -22.565679     -11.710896       0.000000       0.000000  
50%         0.007666      -0.011678       0.041031       0.016587      -0.052790       0.001239       0.011234      22.000000       0.000000  
max        10.503090      22.083545       4.022866       6.070850       3.517346      12.152401      33.847808   11898.090000       1.000000
```

Notes:

We see it the columns are all scaled and numerical.

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

Let’s separate X and y for train and test splits.

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
fit_time       0.016211
score_time     0.001909
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
fit_time       6.238429
score_time     0.008961
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
  - Spotting a class (spot fraud transaction, spot spam, spot disease)

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

<img src="/module7/module7_01/unnamed-chunk-17-1.png" width="50%" />

Notes:

A **confusion matrix** is a table that visualization the performance of
an algorithm. It shows the labels possible and how many of each label
the model predicts correctly and incorrectly.

Once we fit on our training portion, we can use the
`plot_confusion_matrix` function from sklearn.

In this case, we are looking at the validation portion only.

This results in a 4 by 4 matrix with the labels `Non fraud` and `Fraud`
on each axis.

#### Careful:

Scikit-learn’s convention is to have the true label as the rows and the
predicted label as the columns.

Others do it the other way around, e.g., the confusion matrix
<a href=" https://en.wikipedia.org/wiki/Confusion_matrix" target="_blank">Wikipedia
article</a> .

---

<img src="/module7/module7_01/unnamed-chunk-18-1.png" width="50%" style="display: block; margin: auto;" />

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

``` python
from sklearn.model_selection import cross_val_predict
```

``` python
cv_predictions = cross_val_predict(pipe, X_train, y_train)
```

``` python
confusion_matrix(y_train, cv_predictions)
```

```out
array([[139255,     62],
       [    61,    176]])
```

Notes:

You can also calculate confusion matrix with cross-validation using
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html" target="_blank">`cross_val_predict`</a>.

This gives us a prediction for each example but this method does not let
us conveniently use `plot_confusion_matrix`.

---

# Let’s apply what we learned\!

Notes: <br>

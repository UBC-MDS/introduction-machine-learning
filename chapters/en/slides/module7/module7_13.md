---
type: slides
---

# Imbalanced datasets

Notes: <br>

---

### Class imbalance in training sets

``` python
X_train.head()
```

```out
            Time        V1        V2        V3        V4        V5        V6        V7        V8        V9       V10       V11       V12       V13       V14       V15       V16       V17       V18       V19       V20       V21       V22       V23       V24       V25       V26       V27       V28  Amount
121775   76314.0  1.505415 -0.546326 -0.518913 -0.837170 -0.230795 -0.307345 -0.335224 -0.170660 -0.562701  0.599311 -1.857011 -1.303989 -0.410675 -0.150114 -0.056004  0.964810  0.150849 -1.124012  1.343652  0.067662  0.056323  0.172884 -0.337323 -0.969960  0.949936  0.022535 -0.017247 -0.010005   20.00
128746   78823.0 -0.735559  0.459686  2.093094  1.015258  0.159731  0.371070  0.364816 -0.000034 -0.086065  0.126878  1.509075  1.146465  0.156860 -0.325768 -0.323882 -0.749078  0.099186 -0.431016  0.732604 -0.106438 -0.252724 -0.311300  0.202871  0.207971 -0.113774 -0.597624 -0.285776 -0.258780   12.99
59776    48998.0  1.217941  0.783337 -0.070014  1.374358 -0.000839 -1.349727  0.403202 -0.348353 -0.325207 -0.609165  0.582306  0.830314  1.373824 -1.323906  0.874692  0.340441  0.708171  0.180238 -0.715698 -0.022496 -0.013735  0.101074 -0.096465  0.643097  0.697356 -0.352676  0.041878  0.057699    1.00
282774  171138.0  2.024211 -0.586693 -2.554675 -0.837342  2.239626  3.484106 -0.627836  0.827238  0.996288 -0.183073 -0.298045  0.532599 -0.310598  0.254439 -0.016452 -0.724269 -0.129439 -0.559655  0.146863 -0.160179  0.080140  0.431864  0.069908  0.763373  0.234427  0.214823 -0.008147 -0.068130   10.00
268042  163035.0 -0.151161  1.067465 -0.771064  0.138756  1.629341  0.048551  0.996537  0.006544 -0.480459 -1.407786  0.835783  0.572265  0.595428 -2.102961 -0.638511 -0.095413  1.495076  1.310101  1.374670  0.224360 -0.018155  0.125797 -0.202950 -0.113053 -0.209530 -0.209491  0.213933  0.233276   36.26
```

``` python
y_train.value_counts('Class')
```

```out
0    0.998237
1    0.001763
Name: Class, dtype: float64
```

Notes:

A class imbalance typically refers to having many more examples of one
class than another in one’s training set.

We’ve seen this in our fraud dataset where our `class` target column had
many more non-fraud than fraud examples where the classes are
imbalanced.

Real-world data is often imbalanced and can be seen in scenarios such
as:

  - Ad clicking data (Only around \~0.01% of ads are clicked.)
  - Spam classification datasets.

---

### Addressing class imbalance

A very important question to ask yourself: ***“Why do I have a class
imbalance?”***

  - Is it because one class is much rarer than the other?

  - Is it because of my data collection methods?

But, if you answer “no” to both of these, it may be fine to just ignore
the class imbalance.

Notes:

A very important question to ask yourself: ***“Why do I have a class
imbalance?”***

  - Is it because one class is much rarer than the other?
      - If it’s just because one is rarer than the other, you need to
        ask whether you care about False positives or False negatives
        more than the other.  
  - Is it because of my data collection methods?
      - If it’s the data collection, then that means *your test and
        training data come from different distributions*\!

But, if you answer “no” to both of these, it may be fine to just ignore
the class imbalance.

---

### Handling imbalance

There are two common approaches to this:

1.  **Changing the training procedure**

2.  **Changing the data (not in this course)**
    
      - Undersampling
      - Oversampling

Notes:

Can we change the model itself so that it considers the errors that are
important to us?

There are two common approaches to this:

1.  **Changing the training procedure**

2.  **Changing the data (not in this course)**
    
      - Undersampling
      - Oversampling

---

### Changing the training procedure: *class\_weight*

<center>

<img src="/module7/weights-sklearn.png"  width = "80%" alt="404 image" />

</center>

    `class_weight: dict or ‘balanced’, default=None`
    
    Set the parameter C of class i to class_weight[i] * C for SVC. 
    If not given, all classes are supposed to have weight one. 
    The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in 
    the input data as n_samples / (n_classes * np.bincount(y))

Notes:

All `sklearn` classifiers have a parameter called `class_weight`.

This allows you to specify that one class is more important than
another.

For example, maybe a false negative is 10x more problematic than a false
positive.

So, if you look in our example of the SVM classifier, we see
`class_weight` in the documentation.

---

``` python
pipe_tree = make_pipeline((StandardScaler()),
                          (DecisionTreeClassifier(random_state=7)))
pipe_tree.fit(X_train,y_train);
```

``` python
pipe_200 = make_pipeline((StandardScaler()),
                               (DecisionTreeClassifier(random_state=7, class_weight={1:100})))
pipe_200.fit(X_train,y_train);
```

<img src="/module7/module7_13/unnamed-chunk-7-1.png" width="78%" />

Notes: When we made our model before, we can see our confusion matrix on
the left.

Now let’s rebuild our pipeline but using the `class_weight` argument and
setting it as`class_weight={1:200}`.

This is equivalent to saying “repeat every positive example 200x in the
training set”, but repeating data would slow down the code, whereas this
doesn’t.

Notice that we now have reduced false negatives and predicted more Fraud
this time.

But, as a consequence, we are also increasing false positives.

---

## class\_weight=“balanced”

``` python
pipe_balanced = make_pipeline((StandardScaler()),
                               (DecisionTreeClassifier(random_state=7, class_weight="balanced")))
pipe_balanced.fit(X_train,y_train);
```

<img src="/module7/module7_13/unnamed-chunk-9-1.png" width="80%" />

Notes:

We can also set `class_weight="balanced"`.

This sets the weights so that the classes are “equal”.

We have reduced false negatives but we have many more false positives
now\!

---

### Are we doing better with *class\_weight=“balanced”*?

``` python
pipe_tree.score(X_valid, y_valid)
```

```out
0.9989968232737001
```

``` python
pipe_balanced.score(X_valid, y_valid)
```

```out
0.9988129075405451
```

Notes:

Changing the class weight will **generally reduce accuracy**.

The original model was trying to maximize accuracy. Now you’re telling
it to do something different.

But that ok since accuracy isn’t the only metric that matters.

Let’s explain why this happens.

Sincere there are so many more negative examples than positive, false
positives affect accuracy much more than false negatives.

Thus, precision matters a lot more than recall.

So, the default method trades off a lot of recall for a bit of
precision.

---

### Stratified Splits

<center>

<img src="/module7/kfolds.png"  width = "840%" alt="404 image" />

</center>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html" target="_blank">Attribution:
Scikit Learn</a>

Notes:

A similar idea of “balancing” classes can be applied to data splits.

For example, with cross-validation, there is also
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html" target="_blank">`StratifiedKFold`</a>.

From the documentation it says

*“This cross-validation object is a variation of KFold that returns
stratified folds. The folds are made by preserving the percentage of
samples for each class.”*

In other words, if we have 10% negative examples in total, then each
fold will have 10% negative examples.

---

<center>

<img src="/module7/stratified.png"  width = "84%" alt="404 image" />

</center>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.htmll" target="_blank">Attribution:Scikit
Learn</a>

Notes:

We have the same option in `train_test_split` with the `stratify`
argument.

---

### Is stratifying a good idea?

Yes and no:

  - No longer a random sample.
  - It can be especially useful in multi-class situations.

But in general, these are difficult questions to answer.

Notes:

Well, it’s no longer a random sample, which is probably theoretically
bad, but not that big of a deal and If you have many examples, it
shouldn’t matter as much.

It can be especially useful in multi-class situations, say if you have
one class with very few cases.

In general, these are difficult questions to answer.

---

# Let’s apply what we learned\!

Notes: <br>

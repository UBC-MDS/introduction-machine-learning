---
type: slides
---

# Multi-class measurements

Notes: <br>

---

<img src="/module7/module7_09/unnamed-chunk-3-1.png" width="45%" style="display: block; margin: auto;" />

``` python
print(classification_report(y_valid, pipe_tree.predict(X_valid),
        target_names=["non-fraud", "fraud"]))
```

```out
              precision    recall  f1-score   support

   non-fraud       1.00      1.00      1.00     59708
       fraud       0.69      0.75      0.72       102

    accuracy                           1.00     59810
   macro avg       0.85      0.87      0.86     59810
weighted avg       1.00      1.00      1.00     59810
```

Notes:

Right now, we have only seen measurements about target columns with
binary values.

What happens when we have a target with more than 2 classes?

---

``` python
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
```

``` python
digits = load_digits()
digits.images[-1]
```

```out
array([[ 0.,  0., 10., 14.,  8.,  1.,  0.,  0.],
       [ 0.,  2., 16., 14.,  6.,  1.,  0.,  0.],
       [ 0.,  0., 15., 15.,  8., 15.,  0.,  0.],
       [ 0.,  0.,  5., 16., 16., 10.,  0.,  0.],
       [ 0.,  0., 12., 15., 15., 12.,  0.,  0.],
       [ 0.,  4., 16.,  6.,  4., 16.,  6.,  0.],
       [ 0.,  8., 16., 10.,  8., 16.,  8.,  0.],
       [ 0.,  1.,  8., 12., 14., 12.,  1.,  0.]])
```

<img src="/module7/module7_09/unnamed-chunk-7-1.png" width="40%" />

Notes:

This time we are going to look at a dataset of images.

In this case, each image is a hand-written digit (0-9).

The data for a single image is represented by a matrix that is shaped 8
by 8. This corresponds to each pixel of the image.

---

``` python
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(
    digits['data'] / 16., digits['target'], random_state=0)
    
knn = KNeighborsClassifier().fit(X_train_digits, y_train_digits)
pred = knn.predict(X_test_digits)
print("Accuracy: ", accuracy_score(y_test_digits, pred).round(4))
```

```out
Accuracy:  0.98
```

Notes:

We are going to do the same thing we’ve always done and predict the
digit by splitting our data.

In this case, our `X` is the column `data` and our target is the column
`target`.

We use a `KNeighborsClassifier` to fit and predict our accuracy using
`accuracy_score()`.

Here we get an accuracy of 98%.

But what does this mean for our metrics?

---

## Confusion matrix for multi-class

``` python
plot_confusion_matrix(knn, X_test_digits, y_test_digits, cmap='gray_r');
plt.show()
```

<img src="/module7/module7_09/unnamed-chunk-9-1.png" width="55%" />

Notes:

We see that we can still compute a confusion matrix, for problems with
more than 2 labels in the target column.

The diagonal values are the correctly labeled digits and the rest are
the errors.

---

``` python
print(classification_report(y_test_digits, pred, digits=4))
```

```out
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        37
           1     0.9767    0.9767    0.9767        43
           2     1.0000    0.9773    0.9885        44
           3     0.9574    1.0000    0.9783        45
           4     1.0000    0.9737    0.9867        38
           5     0.9592    0.9792    0.9691        48
           6     0.9811    1.0000    0.9905        52
           7     0.9600    1.0000    0.9796        48
           8     1.0000    0.9167    0.9565        48
           9     0.9787    0.9787    0.9787        47

    accuracy                         0.9800       450
   macro avg     0.9813    0.9802    0.9805       450
weighted avg     0.9805    0.9800    0.9799       450
```

Notes:

This time, we have different precision and recall values depending on
which digit we specify as our “positive” label.

Again the `support` column on the right shows the number of examples of
each digit.

What about this `macro avg` and `weight avg` we see on the bottom?

What are these?

---

### Macro average vs weighted average

``` python
print(classification_report(y_test_digits, pred, digits=4))
```

```out
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        37
           1     0.9767    0.9767    0.9767        43
           2     1.0000    0.9773    0.9885        44
           3     0.9574    1.0000    0.9783        45
           4     1.0000    0.9737    0.9867        38
           5     0.9592    0.9792    0.9691        48
           6     0.9811    1.0000    0.9905        52
           7     0.9600    1.0000    0.9796        48
           8     1.0000    0.9167    0.9565        48
           9     0.9787    0.9787    0.9787        47

    accuracy                         0.9800       450
   macro avg     0.9813    0.9802    0.9805       450
weighted avg     0.9805    0.9800    0.9799       450
```

**Macro average:** Give equal importance to all classes.

**Weighted average:** Weighted by the number of samples in each class
and divide by the total number of samples.

Notes:

We saw them before when we were using binary-class problems but these
metrics are more useful when predicting multiple classes.

**Macro average** is useful when you want to give equal importance to
all classes irrespective of the number of instances in each class.

**Weighted average** gives equal importance to all examples. So, when
you care about the overall score and do not care about the score on a
specific class, you could use it.

Which one is relevant, depends upon whether you think each class should
have the same weight or each sample should have the same weight.

---

# Let’s apply what we learned\!

Notes: <br>

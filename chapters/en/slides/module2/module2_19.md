---
type: slides
---

# Generalization

Notes: <br>

---

### Visualizing model complexity using decision boundaries

``` python
classification_df = pd.read_csv("data/quiz2-grade-toy-classification.csv")
classification_df
```

```out
    ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1   quiz2
0               1                 1    92    93    84    91     92      A+
1               1                 0    94    90    80    83     91  not A+
2               0                 0    78    85    83    80     80  not A+
3               0                 1    91    94    92    91     89      A+
4               0                 1    77    83    90    92     85      A+
..            ...               ...   ...   ...   ...   ...    ...     ...
16              0                 0    75    91    93    86     85      A+
17              1                 0    86    89    65    86     87  not A+
18              1                 1    91    93    90    88     82  not A+
19              0                 1    77    94    87    81     89  not A+
20              1                 1    96    92    92    96     87      A+

[21 rows x 8 columns]
```

Notes:

In the last lecture, we learned about decision boundaries.

We saw that we could visualize the splitting of decision trees using
these boundaries.

Let’s use our familiar quiz2 data back again to build on our decision
boundary knowledge.

---

``` python
X = classification_df.drop(columns=["quiz2"])
y = classification_df["quiz2"]
```

``` python
X_subset = X[["lab4", "quiz1"]]
X_subset.head()
```

```out
   lab4  quiz1
0    91     92
1    83     91
2    80     80
3    91     89
4    92     85
```

Notes:

If we subset our data and look at the 2 features from data named `lab4`
and `quiz1` we can see the values the decision tree is splitting on.

---

``` python
depth = 1
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y);
model.score(X_subset, y)
```

```out
0.7142857142857143
```

<center>

<img src="/module2/module2_18a.png"  width = "30%" alt="404 image" />

</center>

Notes:

In the following decision tree model, this decision boundary is created
by asking one question.

---

<center>

<img src="/module2/module2_18a.png"  width = "25%" alt="404 image" />

</center>

<img src="/module2/module2_19/unnamed-chunk-7-1.png" width="65%" style="display: block; margin: auto;" />

Notes:

Here the red region corresponds to the “not A+” class and the blue
region corresponds to the “A+” class.

We can see there is a line separating the red region and the blue region
which is called the **decision boundary** of the model.

---

``` python
depth = 2
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y);
model.score(X_subset, y)
```

```out
0.8095238095238095
```

<center>

<img src="/module2/module2_18b.png"  width = "30%" alt="404 image" />

</center>

Notes:

Let’s see what happens to our decision boundary when we change for
different tree heights.

In the following model, this decision boundaries are created by asking
two questions.

---

<center>

<img src="/module2/module2_18b.png"  width = "25%" alt="404 image" />

</center>

<img src="/module2/module2_19/unnamed-chunk-10-1.png" width="65%" style="display: block; margin: auto;" />

---

``` python
depth = 4
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y);
model.score(X_subset, y)
```

```out
0.9523809523809523
```

<center>

<img src="/module2/module2_18c.png"  width = "40%" alt="404 image" />

</center>

Notes:

In the next model, this decision boundaries are created by asking three
questions.

---

``` python
model.score(X_subset, y)
```

```out
0.9523809523809523
```

<img src="/module2/module2_19/unnamed-chunk-14-1.png" width="70%" style="display: block; margin: auto;" />

Notes:

---

``` python
depth = 10
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y);
model.score(X_subset, y)
```

```out
1.0
```

<center>

<img src="/module2/module2_18d.png"  width = "45%" alt="404 image" />

</center>

Notes:

For this last model, the decision boundaries are created by asking 10
questions.

---

``` python
model.score(X_subset, y)
```

```out
1.0
```

<img src="/module2/module2_19/unnamed-chunk-18-1.png" width="65%" style="display: block; margin: auto;" />

Notes:

Our model has 0% error\!\! But it’s also becoming more and more specific
and sensitive to the training data. Is it good or bad?

---

## Fundamental goal of machine learning

<center>

<img src="/module2/generalization-train.png" width = "50%" alt="404 image" />

</center>

Notes:

The fundamental goal of machine learning is **to generalize beyond what
we see in the training examples**.

Example: Imagine that a learner sees the following images and
corresponding labels.

---

### Generalizing to unseen data

<center>

<img src="/module2/generalization-predict.png" width = "100%" alt="404 image" />

</center>

Notes:

Now the learner is presented with new images (1 to 4) for prediction.

What prediction would you expect for each image?

We want the learner to be able to generalize beyond what it has seen in
the training data, but these new examples should be representative of
the training data.

For instance, is it fair to expect the learner to label image 4
correctly?

---

## Training score versus Generalization score

  - Given a model in ML, people usually talk about two kinds of
    accuracies (scores):

<!-- end list -->

1.  Accuracy on the training data

2.  Accuracy on the entire distribution of data

Notes:

This is where the model accuracy comes in.

People usually talk about two kinds of accuracies (scores) in machine
learning:

1.  Accuracy on the training data

2.  Accuracy on the entire distribution of data

We are interested in the accuracy on the entire distribution, but we do
not have access to the entire distribution\!

What do we do?

We will cover this, in the next module.

---

# Let’s apply what we learned\!

Notes: <br>

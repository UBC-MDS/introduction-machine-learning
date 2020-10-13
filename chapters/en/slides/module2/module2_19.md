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
X = classification_df.drop(["quiz2"], axis=1)
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
model.fit(X_subset, y)
```

<center>

<img src="/module2/module2_18a.png"  width = "40%" alt="404 image" />

</center>

Notes:

In the following decision tree model, this decision boundary is created
by asking one question.

---

<img src="/module2/module2_19/unnamed-chunk-7-1.png" width="672" />

Notes:

Here the red region corresponds to the “not A+” class and the blue
region corresponds to the “A+” class.

We can see there is a line separating the red region and the blue region
which is called the **decision boundary** of the model.

---

``` python
depth = 2
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y)
```

<center>

<img src="/module2/module2_18b.png"  width = "50%" alt="404 image" />

</center>

Notes:

Let’s see what happens to our decision boundary when we change for
different tree heights.

In the following model, this decision boundaries are created by asking
two questions.

---

<img src="/module2/module2_19/unnamed-chunk-10-1.png" width="672" />

---

``` python
depth = 3
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y)
```

<center>

<img src="/module2/module2_18c.png"  width = "60%" alt="404 image" />

</center>

Notes:

In the next model, this decision boundaries are created by asking three
questions.

---

<img src="/module2/module2_19/unnamed-chunk-13-1.png" width="672" />

Notes:

---

``` python
depth = 10
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y)
```

<center>

<img src="/module2/module2_18d.png"  width = "45%" alt="404 image" />

</center>

Notes:

For this last model, the decision boundaries are created by asking 10
questions.

---

<img src="/module2/module2_19/unnamed-chunk-16-1.png" width="672" />

Notes:

Our model is becoming more and more specific and sensitive to the
training data.

Is this a good thing or a bad thing?

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

## Training error versus Generalization error

  - Given a model 𝑀, in ML, people usually talk about two kinds of
    errors of 𝑀:

<!-- end list -->

1.  Error on the training data
    <img src="/module2/trainning_e.gif"  width = "11%" alt="404 image" />

2.  Error on the entire distribution 𝐷 of data
    <img src="/module2/d_e.gif"  width = "9%" alt="404 image" />

Notes:

This is where the idea of error comes in.

People usually talk about two kinds of errors in a model.

1.  Error on the training data: \(error_{training}(M)\)
2.  Error on the entire distribution \(D\) of data: \(error_{D}(M)\)

We are interested in the error on the entire distribution, but we do not
have access to the entire distribution\!

What do we do?

We will cover this, in the next module.

---

# Let’s apply what we learned\!

Notes: <br>

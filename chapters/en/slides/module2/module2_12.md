---
type: slides
---

# Parameters and hyperparameters

Notes: <br>

---

<center>

<img src="/module2/valves.jpg"  width = "80%" alt="404 image" />

</center>

  - ***Parameters***: Derived during training
  - ***Hyperparameters***: Adjustable parameters that can be set before
    training.

Notes:

When you call `fit`, a bunch of values get set, like the split variables
and split thresholds.

  - These are called **parameters**.

But even before calling `fit` on a specific data set, we can set some
“knobs” that control the learning.

  - These are called **hyperparameters**.

---

``` python
classification_df = pd.read_csv("data/quiz2-grade-toy-classification.csv")
classification_df.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1   quiz2
0              1                 1    92    93    84    91     92      A+
1              1                 0    94    90    80    83     91  not A+
2              0                 0    78    85    83    80     80  not A+
3              0                 1    91    94    92    91     89      A+
4              0                 1    77    83    90    92     85      A+
```

``` python
X = classification_df.drop(["quiz2"], axis=1)
y = classification_df["quiz2"]
```

``` python
model = DecisionTreeClassifier(max_depth=1)  
model.fit(X, y)
```

Notes:

In scikit-learn, hyperparameters are set in the constructor.

`max_depth`is a hyperparameter that lets us decide and set the maximum
depth of the decision tree.

We can set the argument `max_depth=1` in our code so that it builds a
***decision stump***.

---

<center>

<img src="/module2/module2_12a.png"  width = "40%" alt="404 image" />

</center>

Notes:

---

``` python
model.score(X, y)
```

```out
0.7619047619047619
```

``` python
model2 = DecisionTreeClassifier(max_depth=2)  
model2.fit(X, y)
```

``` python
model2.score(X, y)
```

```out
0.8571428571428571
```

``` python
model3 = DecisionTreeClassifier(max_depth=4)  
model3.fit(X, y)
```

``` python
model3.score(X, y)
```

```out
0.9523809523809523
```

Notes:

How well does our model score when using setting `max_depth=1`?

Ok, 76% that’s not too bad but what happens when we increase the
`max_depth` to 2?

It looks like it’s increasing\!

Increasing `max_depth` to 4 ,makes the accuracy increase to 95%.

We can now conclude that as `max_depth` increases, the accuracy of the
training data does as well.

We will introduce in the last slide deck of this module, why having
perfect accuracy isn’t always the best idea.

---

``` python
model4 = DecisionTreeClassifier(min_samples_split=2)  
model4.fit(X, y)
```

Notes:

Let’s explore another different hyperparameter `min_samples_split`.

`min_samples_split` sets the minimum number of samples required to split
an internal node.

Remember our decision boundaries? This will set a minimum number of
observations that need to be on either side of the boundary.

---

<center>

<img src="/module2/module2_12b.png"  width = "50%" alt="404 image" />

</center>

Notes:

---

``` python
model4.score(X, y)
```

```out
1.0
```

``` python
model5 = DecisionTreeClassifier(min_samples_split=4) 
model5.fit(X, y)
```

``` python
model5.score(X,y)
```

```out
0.9523809523809523
```

``` python
model6 = DecisionTreeClassifier(min_samples_split=10) 
model6.fit(X, y)
```

``` python
model6.score(X,y)
```

```out
0.9047619047619048
```

Notes:

In this case, as the value of the hyperparameter `min_samples_split`
increases, the accuracy decreases.

It’s really important to take into consideration that this accuracy is
referring to the accuracy of predictions on the same data that we
trained our model on.

---

<center>

<img src="/module2/decisiontree.png"  width = "80%" alt="404 image" />

</center>

<br> See this link
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" target="_blank">here</a>
.

Notes:

There are many other hyperparameters for decision trees you can explore
at the link
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" target="_blank">here</a>
.

---

### To summarize

  - **parameters** are automatically learned by the algorithm during
    training
  - **hyperparameters** are specified based on:
      - expert knowledge
      - heuristics, or
      - systematic/automated optimization (more on that in the upcoming
        modules)

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# Decision Tree Classifiers

Notes: <br>

---

<center>

<img src="/module2/lingo_tree.png"  width = "100%" alt="404 image">

</center>

Notes:

Now that we know the structure of a decision tree, let’s build a
decision tree model.

---

``` python
from sklearn.tree import DecisionTreeClassifier
```

``` python
model = DecisionTreeClassifier()
```

``` python
new_example
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 0     1     1     0     0      0
```

``` python
model.predict(new_example)
```

``` out
NotFittedError: This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

Detailed traceback: 
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.8/site-packages/sklearn/tree/_classes.py", line 426, in predict
    check_is_fitted(self)
  File "/usr/local/lib/python3.8/site-packages/sklearn/utils/validation.py", line 72, in inner_f
    return f(**kwargs)
  File "/usr/local/lib/python3.8/site-packages/sklearn/utils/validation.py", line 1019, in check_is_fitted
    raise NotFittedError(msg % {'name': type(estimator).__name__})
```

Notes:

We can use the same steps we used when building a baseline model, to
build a decision tree model with `scikit-learn`.

Recall that `scikit-learn` uses the term `fit` for training or learning
and uses `predict` for prediction.

We must import the function from the `sklearn.tree` library and then we
can create a decision tree using the `DecisionTreeClassifier()`
function.

Is our new model ready to predict now that we have made our decision
tree model?

Let’s see what happens when we try to predict the quiz2 mark of a new
example.

We get an error\! We forgot the crucial step of fitting our model.

---

``` python
X_binary.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1     1     1     0     1      1
1              1                 0     1     1     0     0      1
2              0                 0     0     0     0     0      0
3              0                 1     1     1     1     1      0
4              0                 1     0     0     1     1      0
```

``` python
y.head()
```

```out
0        A+
1    not A+
2    not A+
3        A+
4        A+
Name: quiz2, dtype: object
```

``` python
model.fit(X_binary, y)
```

Notes:

We need to make sure that we `.fit()` our model before we `.predict()`.

In the decision tree algorithm, the fitting stage is where the model
learns about the data and sets the *if and else* statements.

---

<center>

<img src="/module2/dt_quiz2.png"  width = "80%" alt="404 image" />

</center>

Notes:

Let’s take a look at what our tree looks like. We can see all the nodes
of the tree and what the root condition is.

---

``` python
new_example
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 0     1     1     0     0      0
```

``` python
print("Prediction for example:" + (model.predict(new_example)[0]))
```

```out
Prediction for example:not A+
```

Notes:

Now that the model is fitted, we will be able to predict using the built
model.

---

``` python
model.score(X_binary, y)
```

```out
0.9047619047619048
```

Notes:

We can check how accurate our model is using `.score()`.

This model predicts on data it’s already seen with 90% accuracy.

---

### How does predict work?

``` python
observation
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 0     1     1     0     1      0
```

<center>

<img src="/module2/predict.gif"  width = "70%" alt="404 image" />

</center>

Notes:

Let’s discuss how predict works.

We have a learned tree and a test example.

Let’s start at the top of the tree and ask binary questions at each node
and follow the appropriate path in the tree.

Once you are at a leaf node, you have the prediction.

The model only considers the features which are in the learned tree and
ignores all other features.

---

### How does fit work

  - Which features are most useful for classification?
  - Minimize **impurity** at each question
  - Common criteria to minimize impurity
      - Gini Index
      - Information gain
      - Cross entropy

Notes:

We aren’t going to go into this in much detail, but the fitting of a
decision tree has a lot to do with important features (which columns
contribute the most to the decision-making process) and minimizing the
“impurity”.

We don’t need to worry about this for this course.

---

# Let’s apply what we learned\!

Notes: <br>

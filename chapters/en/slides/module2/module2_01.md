---
type: slides
---

# Decision Tree Classifiers

Notes: <br>

---

### Recap

Examples:

<center>

<img src="/module2/quiz2-grade-toy.png"  width = "85%" alt="404 image" />

</center>

Notes:

So we have built a baseline model but can we do better than that?

If you are asked to write a program to predict whether a student gets an
A+ or not in quiz2, how would you go for it?

Before we go forward let’s try to make things a little more simple and
binarize the input data.

---

## A program for prediction using a set of rules with *if else* statements

<center>

<img src="/module2/quiz2-grade-toy.png" height="600" width="600">

</center>

  - How about a rule-based algorithm with a number of *if else*
    statements?  

<!-- end list -->

    if class attendance == 1 and quiz1 == 1:
        quiz2 == "A+"
    elif class attendance == 1 and lab3 == 1 and lab4 == 1:
        quiz2 == "A+"
    ...

Notes:

Now that we have a model where our features are 1 of 2 options, how
about we have a rule-based algorithm with a number of *if else*
statements?  
We learned about conditions and *if else* statements in module 5 of
***Programming in Python for Data Science***, now let’s incorporate this
concept into a machine learning problem. For example:

    if class attendance == 1 and quiz1 == 1:
        quiz2 == "A+"
    elif class attendance == 1 and lab3 == 1 and lab4 == 1:
        quiz2 == "A+"
    ...

  - How many rules do we need?
  - How many possible rule combinations could there be given 7 binary
    features?
      - Gets unwieldy pretty quickly

---

``` python
X.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
1              1                 0    94    90    80    83     91
2              0                 0    78    85    83    80     80
3              0                 1    91    94    92    91     89
4              0                 1    77    83    90    92     85
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

Notes:

Let’s take our 𝑋 and 𝑦 from our quiz2 data that we had before.

---

``` python
X_binary = X.copy()
columns = ["lab1", "lab2", "lab3", "lab4", "quiz1"]
for col in columns:
    X_binary[col] = X_binary[col].apply(
        lambda x: 1 if x >= 90 else 0)
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

Notes:

Now let’s binarize the features in `X` like we discussed. Now we can see
that each column has only a value of either `0` or `1`.

Now we have our data in a perferred way, how do we make predictions with
the `if else` statements we talked about?

---

## Decision trees

<center>

<img src="/module2/nature.png"  width = "85%" alt="404 image" />

</center>

Notes:

The decision tree models use an algorithm that derives such rules from
data in a principled way.

---

## Decision trees Terminology

<center>

<img src="/module2/lingo_tree.png"  width = "85%" alt="404 image">

</center>

Note:

A tree is a type of structure with branches and nodes that is an
effective way to visualize the process of decision making.

A tree starts at the top of the tree which is described as the
***root***.

Each decision is called a ***node*** and they are connected by
***branches***.

With the decision tree algorithm in machine learning, the tree can have
at most 2 **nodes** resulting from it, also know as **children**.

A Decision Tree that only results in 2 children for each node takes on a
specific named called a **Binary Decision Tree**.

The maximum depth of a tree is somewhat like the “height” or how “tall”
a tree stands. It refers to the the length of the longest path from a
root to a leaf.

---

<center>

<img src="/module2/example3.png"  width = "85%" alt="404 image">

</center>

Note:

Using our quiz2 dataset as an example a tree may look something like
this.

This tree has a depth of 3.

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

Before looking into the algorithm, let’s `fit` and `predict` with a
decision tree built using `scikit-learn`.

Recall that `scikit-learn` uses the term `fit` for training or learning
and uses `predict` for prediction.

We must import the function from the `sklearn.tree` library and then we
can create a decision tree using the `DecisionTreeClassifier()`
function.

Is our new model ready to predict now that we’ve made our decision tree
model?

Let’s see what happens when we try to predict the quiz2 mark of a new
example.

We get an error\! We forgot the crucial step of fitting out model.

---

``` python
model.fit(X_binary, y)
```

```out
DecisionTreeClassifier()
```

Notes:

We need to make sure that we `.fit()` our model before we `.predict()`.
In the decision tree algorithm, the fitting stage is where the model
learns about the data and sets the *if and else* statements.

---

<center>

<img src="/module2/dt_quiz2.png"  width = "45%" alt="404 image" />

</center>

Notes:

let’s take a look at what our tree looks like. We can see all the nodes
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
print("Prediction for example: %s" % (model.predict(new_example)[0]))
```

```out
Prediction for example: not A+
```

Notes:

Now that the model is fitted we’ll be able to predict using the built
model.

---

### How does predict work?

<center>

<img src="/module2/predict.gif"  width = "45%" alt="404 image" />

</center>

Notes:

Let’s discuss how predict really works.

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

We arn’t going to go into this with much detail but the fitting of a
decision trees has a lot to do with important features (which columns
contribute the most to the decision making process) and minimizing the
“impurity”.

We don’t need to worry about this for this course.

---

Notes:

---

# Let’s apply what we learned\!

Notes: <br>

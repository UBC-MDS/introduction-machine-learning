---
type: slides
---

# Decision Tree Classifiers

Notes: <br>

---

## Improving the baseline model

Examples:

<center>

<img src="/module2/quiz2-grade-toy.png"  width = "85%" alt="404 image" />

</center>

Notes:

So we have built a baseline model, but can we do better than that?

If you are asked to write a program to predict whether a student gets an
A+ or not in quiz2, how would you go for it?

Before we go forward, let’s try to make things a little simpler and
binarize the input data.

---

## A program for prediction using a set of rules with *if/else* statements

<center>

<img src="/module2/quiz2-grade-toy.png" height="600" width="600">

</center>

  - How about a rule-based algorithm with several *if/else*
    statements?  

<!-- end list -->

    if class attendance == 1 and quiz1 == 1:
        quiz2 == "A+"
    elif class attendance == 1 and lab3 == 1 and lab4 == 1:
        quiz2 == "A+"
    ...

Notes:

Now that we have a model where our features are 1 of 2 options, how
about we have a rule-based algorithm with several *if/else*
statements?  
We learned about conditions and *if/else* statements in module 5 of
***Programming in Python for Data Science***, now let’s incorporate this
concept into a machine learning problem. For example:

    if class attendance == 1 and quiz1 == 1:
        quiz2 == "A+"
    elif class attendance == 1 and lab3 == 1 and lab4 == 1:
        quiz2 == "A+"
    ...

  - How many rules do we need?
  - How many possible rule combinations could there be, given the
    existing 7 binary features? It can get unwieldy pretty quickly.

Decision tree gives us a better way to do this\!

---

``` python
classification_df = pd.read_csv("data/quiz2-grade-toy-classification.csv")
classification_df.head(3)
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1   quiz2
0              1                 1    92    93    84    91     92      A+
1              1                 0    94    90    80    83     91  not A+
2              0                 0    78    85    83    80     80  not A+
```

``` python
X = classification_df.drop(["quiz2"], axis=1)
X.head(3)
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
1              1                 0    94    90    80    83     91
2              0                 0    78    85    83    80     80
```

``` python
y = classification_df["quiz2"]
y.head(3)
```

```out
0        A+
1    not A+
2    not A+
Name: quiz2, dtype: object
```

Notes:

Let’s take our 𝑋 and 𝑦 from our `quiz2` data that we had before.

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

Now let’s binarize the features in `X` as we discussed. We can see that
each column now only has a value of either `0` or `1`.

Now we have our data in a preferred way, how do we make predictions with
the `if`/`else` statements we talked about?

---

## Decision trees

<center>

<img src="/module2/nature.png"  width = "85%" alt="404 image" />

</center>

Notes:

The decision tree models use an algorithm that derives such rules from
data in a principled way.

---

## Decision trees terminology

<center>

<img src="/module2/lingo_tree.png"  width = "85%" alt="404 image">

</center>

Note:

A tree is a type of structure with branches and nodes that is an
effective way to visualize the process of decision making.

A tree begins at the top which is called the ***root***.

Each decision is called a ***node*** and they are connected by
***branches***.

With the decision tree algorithm in machine learning, the tree can have
at most two **nodes** resulting from it, also known as **children**.

A Decision Tree that only results in 2 children for each node takes on a
specific named called a **Binary Decision Tree**.

The maximum depth of a tree is somewhat like the “height” or how “tall”
a tree stands. It refers to the length of the longest path from a root
to a leaf.

---

<center>

<img src="/module2/example3.png"  width = "85%" alt="404 image">

</center>

Note:

Using our quiz2 dataset as an example, a tree may look something like
this.

This tree has a depth of 3.

---

# Let’s apply what we learned\!

Notes: <br>

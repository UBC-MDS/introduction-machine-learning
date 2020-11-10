---
type: slides
---

# Baselines: Training a Model using Scikit-learn

Notes: <br>

---

### Supervised Learning (Reminder)

  - Tabular data → Machine learning algorithm → ML model → new examples
    → predictions

<center>

<img src="/module1/sup-learning.png"  alt="A caption" width="80%" />

</center>

Notes:

Just to recap what we know, we take tabular data and a machine learning
algorithm and produce a machine learning model.

We can then take new examples and make predictions on them using this
model.

---

### Building a simplest machine learning model using sklearn

<br> <br> <br> Baseline model: **most frequent baseline**: always
predicts the most frequent label in the training set.

Notes:

Let’s build a ***baseline*** simple machine learning algorithm based on
simple rules of thumb.

We are going to build a most frequent baseline model which always
predicts the most frequent label in the training set.

Baselines provide a way to sanity check your machine learning model.

---

## Data

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

Notes:

Let’s take our data.

For this example, we are going to be working with the quiz2
classification data that we have seen previously.

---

## 1\. Create 𝑋 and 𝑦

𝑋 → Feature vectors <br> 𝑦 → Target

``` python
X = classification_df.drop(columns=["quiz2"])
y = classification_df["quiz2"]
```

Notes:

Our first step in building our model is spliting up our tabular data
into the features and the target, also known as 𝑋 and 𝑦.

𝑋 is all of our features in our data, which we also call our ***feature
table***. 𝑦 is our target, which is what we are predicting.

For this problem, all the columns in our dataframe except `quiz2` make
up our 𝑋 and the `quiz2` column, which is our target make up our 𝑦.

<br>

---

## 2\. Create a classifier object

  - `import` the appropriate classifier.
  - Create an object of the classifier.

<!-- end list -->

``` python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
```

Notes:

In order to make our baseline model, we need to import the necessary
library.

We spoke about the Scikit Learn package in the last slide deck.

Here we are importing `DummyClassifier()` which will be used to create
our baseline model.

We specify in the `strategy` argument `most_frequent` which means our
model will always predicts the most frequent label in the training set.

Here we are naming our model `dummy_clf`.

---

## 3\. Fit the classifier

``` python
dummy_clf.fit(X, y)
```

Notes:

Once we have picked and named our model, we give it data to train on.

The model’s “learning” is carried out when we call `fit` on the
classifier object.

In a lot of models, the fitting (also know as the training) stage takes
the longest and is where most of the work occurs. This isn’t always the
case but it is in a lot of them.

---

## 4\. Predict the target of given examples

We can predict the target of examples by calling `predict` on the
classifier object.

Let’s see what it predicts for a single observation first:

``` python
single_obs = X.loc[[0]]
single_obs
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
```

``` python
dummy_clf.predict(single_obs)
```

```out
array(['not A+'], dtype='<U6')
```

Notes:

Now that our model has been trained on existing data, we can predict the
targets.

We are going to try to predict on data that the model has already seen.
In this case `X`, was used to in the fitting stage. This will change
very soon.

Let’s first see what the model predicted for a single observation.

We can see here, that for observation 0, it’s predicting a value of `not
A+`.

This was the most frequent `quiz2` value in the data that we gave it
during the `.fit()` stage.

---

``` python
X
```

```out
    ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0               1                 1    92    93    84    91     92
1               1                 0    94    90    80    83     91
2               0                 0    78    85    83    80     80
3               0                 1    91    94    92    91     89
4               0                 1    77    83    90    92     85
..            ...               ...   ...   ...   ...   ...    ...
16              0                 0    75    91    93    86     85
17              1                 0    86    89    65    86     87
18              1                 1    91    93    90    88     82
19              0                 1    77    94    87    81     89
20              1                 1    96    92    92    96     87

[21 rows x 7 columns]
```

``` python
dummy_clf.predict(X)
```

```out
array(['not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+'], dtype='<U6')
```

Notes:

And if we see the predictions for all the observations in `X`, the model
predicts a value of `not A+` for each one.

We will talk more about `.fit()` and `.predict()` in the next module.

---

## 5\. Scoring your model

In the classification setting, `.score()` gives the accuracy of the
model, i.e., proportion of correctly predicted observations.

<center>

<img src="/module1/predit_total.gif" >

</center>

Sometimes you will also see people reporting error, which is usually
1−𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦

<center>

<img src="/module1/error.gif" >

</center>

``` python
print("The accuracy of the model on the training data:", (dummy_clf.score(X, y).round(3)))
```

```out
The accuracy of the model on the training data: 0.524
```

``` python
print("The error of the model on the training data:", (1 - dummy_clf.score(X, y)).round(3))
```

```out
The error of the model on the training data: 0.476
```

Notes:

Its at this point where we can see how well our baseline model predicts
the `quiz2` value.

In ML models, very often it is not possible to get 100% accuracy. How do
you check how well your model is doing?

In the classification setting, `score()` gives the accuracy of the
model, i.e., proportion of correctly predicted.

Sometimes you will also see people reporting error, which is usually 1 -
accuracy.

We can see that our model’s accuracy on our quiz2 problem is 0.524.

We could also say the error is 0.476.

---

## fit and predict paradigms

The general pattern when we build ML models using `sklearn`:

1.  Creating your 𝑋 and 𝑦 objects
2.  `clf = DummyClassifier()` → create a model (here we are naming it
    `clf`)  
3.  `clf.fit(X, y)` → train the model
4.  `clf.score(X, y)` → assess the model
5.  `clf.predict(Xnew)` → predict on some new data using the trained
    model

Notes:

When building models, there is a general pattern that we repeat.

1.  Creating your 𝑋 and 𝑦 objects
2.  `clf` → create a model (here we are naming it `clf`)  
3.  `clf.fit(X, y)` → train the model
4.  `clf.predict(X)` → predict using the trained model
5.  `clf.score(X, y)` → assess the model

---

# Let’s apply what we learned\!

Notes: <br>

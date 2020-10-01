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

<img src="/module1/supervised-learning.png" height="1200" width="1200">

</center>

Notes:

<br>

---

### Building a simplest machine learning model using sklearn

<br> <br> <br> Baseline models:

  - **uniform baseline**: generate predictions uniformly at random.
  - **most frequent baseline**: always predicts the most frequent label
    in the training set.

Notes:

Let’s build a “baseline” simple machine learning algorithm based on
simple rules of thumb.

For example we can build the following:

  - A uniform baseline model: This generate predictions uniformly at
    random.
  - A most frequent baseline model: This always predicts the most
    frequent label in the training set.

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

Let’s take our data. We are going to be working with the quiz2
classification data that we have seen previously.

---

## 1\. Create 𝑋 and 𝑦

𝑋 → Feature vectors <br> 𝑦 → Target

``` python
X = classification_df.drop(["quiz2"], axis=1)
y = classification_df["quiz2"]
```

Notes:

Our first step in building our model is spliting up our data into the
features and the target, also known as 𝑋 and 𝑦.

X is all of our features in our data, this is called our ***Feature
vectors***. y is our target, what we are predicting.

For this problem, all the columns in our dataframe except `quiz2` make
up our 𝑋 and the `quiz2` column, which is our target make up our 𝑦.

<br>

---

## 2\. Create a classifier or a regressor object

  - `import` the appropriate classifier or regressor.
  - Create an object of the classifier or regressor.

<!-- end list -->

``` python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
```

Notes:

In order to make our baseline model, we need to import the necessary
library.

We spoke about the Scikit Learn package in the last slide deck.

Here we are importing the function `DummyClassifier()` which will be
used to create our baseline model.

We specify in the `strategy` argument `most_frequent` which means our
model will always predicts the most frequent label in the training set.

Here we are naming our model `dummy_clf`.

---

## 3\. Fit the classifier

``` python
dummy_clf.fit(X, y)
```

```out
DummyClassifier(strategy='most_frequent')
```

Notes:

Once we have picked and named our model, we give it data to train on.

The model’s “learning” is carried out when we call `fit` on the
classifier object.

We can see that it returns the model’s specifications as an output. This
output isn’t that important to our analysis and is generally ignored.

In a lot of models, the fitting (also know as the training) stage takes
the longest and is where most of the work occurs. This isn’t always the
case but it is in a lot of them.

---

## 4\. Predict the target of given examples

We can predict the target of examples by calling `predict` on the
classifier object.

``` python
dummy_clf.predict(X)
```

```out
array(['not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+', 'not A+'], dtype='<U6')
```

Notes:

Now that our model has been train on existing data, we can predict the
target of examples by calling `predict` on the classifier object.

It’s at this stage, unlike in `.fit()` where the output is important to
us. It tells us what the model predicts for the observations.

We can see here, that for each observation it’s predicting a value of
`not A+` which was the most frequent `quiz2` value in the data we gave
it during the `fit()` stage.

We will talk more about `.fit()` and `.predict()` in the next module.

---

## 5\. Scoring your model

In the classification setting, the `score()` function gives the accuracy
of the model, i.e., proportion of correctly predicted observations.

<center>

<img src="/module1/predit_total.gif" >

</center>

Sometimes you will also see people reporting error, which is usually
1−𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦

<center>

<img src="/module1/error.gif" >

</center>

``` python
print("The accuracy of the model on the training data: %0.3f" %(dummy_clf.score(X, y)))
```

```out
The accuracy of the model on the training data: 0.524
```

``` python
print("The error of the model on the training data: %0.3f" %(1 - dummy_clf.score(X, y)))
```

```out
The error of the model on the training data: 0.476
```

Notes:

Its at this point where we can see how well our baseline model predicts
the `quiz2` value.

In ML models, very often it is not possible to get 100% accuracy. How do
you check how well your model is doing?

In the classification setting, the `score()` function gives the accuracy
of the model, i.e., proportion of correctly predicted.

Sometimes you will also see people reporting error, which is usually 1 -
accuracy.

We can see that our model’s accuracy on our quiz2 problem is 0.524 which
means the error is 0.476.

---

## fit and predict paradigms

The general pattern when we build ML models using `sklearn`:

1.  Creating your 𝑋 and 𝑦 ojects
2.  `clf` → create a model (here we are naming it `clf`)  
3.  `clf.fit(X, y)` → train the model
4.  `clf.predict(X)` → predict using the trained model
5.  `clf.score(X, y)` → assess the model

Notes:

When building models, there is a general pattern that we repeat.

---

# Let’s apply what we learned\!

Notes: <br>

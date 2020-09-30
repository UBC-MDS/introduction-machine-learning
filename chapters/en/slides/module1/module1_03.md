---
type: slides
---

# Types of machine learning

Notes: <br>

---

Typical learning problems:

  - **Supervised learning (this course)**
    (<a href="https://support.google.com/a/answer/2368132?hl=en" target="_blank">Gmail
    spam filtering</a>)
      - Training a model from input data and its corresponding labels to
        predict new examples.  
  - Unsupervised learning
    (<a href="https://news.google.com/" target="_blank">Google News</a>)
      - Training a model to find patterns in a dataset, typically an
        unlabeled dataset.
  - Reinforcement learning
    (<a href="https://deepmind.com/research/case-studies/alphago-the-story-so-far" target="_blank">AlphaGo</a>)
      - A family of algorithms for finding suitable actions to take in a
        given situation in order to maximize a reward.
  - Recommendation systems
    (<a href="https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf" target="_blank">Amazon
    item recommendation system</a>)
      - Predict the “rating” or “preference” a user would give to an
        item.

Notes:

There are different types of machine learning problems and each type
uses different models.

---

## Supervised learning

<center>

<img src="/module1/supervised-learning.png" height="1000" width="1000">

</center>

Notes:

  - Training data comprises a set of observations (\(X\)) and their
    corresponding targets (\(y\)).
  - We wish to find a model function \(f\) that relates \(X\) to \(y\).
  - Then use that model function to predict the labels of new examples.

---

The example we saw of classifying images is an example of supervised
learning:

``` python
from toy_classifier import classify_image

img = Image.open("alpaca.jpg")
img
```

``` out
```

<img src="/module1/alpaca.jpg" alt="This image is in /static" width="20%">

Notes:

The goal of supervised learning is to use data to find a model that
outputs the right label based on the features.

---

<img src="/module1/alpaca.jpg" alt="This image is in /static" width="20%">

``` python
classify_image(img, 3)
```

```out
Class, Probability
llama, 0.72
Eskimo dog, husky, 0.06
Norwich terrier, 0.05
```

Notes: <br>

---

## Unsupervised learning

<center>

<img src="/module1/unsup-learning.png" alt="" height="900" width="900">

</center>

Notes:

  - Training data consists of observations (\(X\)) without any
    corresponding targets.
  - Unsupervised learning could be used to group similar things together
    in \(X\).

In unsupervised learning, We have data, but no associated response. We
will not be covering unsupervised learning in this course and instead
put our main focus on supervised.

---

## Machine Learning Libraries

<br>

### <a href="https://scikit-learn.org/stable/index.html" target="_blank">scikit-learn</a>

<center>

<img src="/module1/sk-learn.png" alt="" height="900" width="900">

</center>

Notes:

There are a number of machine learning libraries available but for this
course we will be using the `sklearn` library, which is a popular (41.6k
stars on Github) Machine Learning library for Python.

---

## What we know so far…

  - Machine learning is a different paradigm for problem solving.
      - It lets you solve problems that seem “untractable” or
        “unprogrammable”.
      - It reduces the time you spend programming and helps customizing
        and scaling your products.  
  - In supervised learning we are given a set of observations (\(X\))
    and their corresponding targets (\(y\)) and we wish to find a model
    function \(f\) that relates \(X\) to \(y\).
  - In unsupervised learning, we are given a set of observations (\(X\))
    and we wish to group similar things together in \(X\).

Notes:

Here is what we know so far.

---

# Let’s apply what we learned\!

Notes: <br>

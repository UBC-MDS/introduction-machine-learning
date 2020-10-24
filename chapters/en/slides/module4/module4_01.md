---
type: slides
---

# Terminology with analogy-based models

Notes: <br>

---

## Analogy-based models

<br> <br>

<img src='/module4/knn-motivation.png' width="100%">
<a href="https://vipl.ict.ac.cn/en/database.php" target="_blank">Attribution</a>

Notes:

Suppose you are given the following training examples with corresponding
labels and are asked to label a given test example.

An intuitive way to classify the test example is by finding the most
“similar” example(s) from the training set and using that label for
the test example.

---

## Analogy-based algorithms in practice

  - <a href="https://www.hertasecurity.com/en" target="_blank">Herta’s
    High-tech Facial Recognition</a>

<center>

<img src="/module4/face_rec.png"  width = "20%" alt="404 image" />

</center>

  - Recommendation systems

<center>

<img src="/module4/book_rec.png"  width = "90%" alt="404 image" />

</center>

Notes:

Examples of Analogy-based algorithms include:

  - <a href="https://www.hertasecurity.com/en" target="_blank">Herta’s
    High-tech Facial Recognition</a>
      - Feature vectors for human faces
      - 𝑘-NN to identify which face is on their watch list
  - Recommendation systems

---

### Geometric view of tabular data and dimensions

<center>

<img src="/module4/3d-table.png"  width = "100%" alt="404 image" />

</center>

Notes:

To understand analogy-based algorithms it’s useful to think of data as
points in a high dimensional space.

  - Our `X` represents the problem in terms of relevant **features**
    with one dimension for each **feature** (column).

  - Examples are **points in a number-of-features-dimensional space**.

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
train_df.head()
```

```out
     longitude  latitude country
160   -76.4813   44.2307  Canada
127   -81.2496   42.9837  Canada
169   -66.0580   45.2788  Canada
188   -73.2533   45.3057  Canada
187   -67.9245   47.1652  Canada
```

Notes:

Let’s look at our Canadian and United States cities.

How many dimensions (features) are there in this cities data?

If you said 2, then you are off to a good start.

---

``` python
cities_plot = alt.Chart(train_df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue']))
)
cities_plot
```
<img src="/module4/cities_plot.png" alt="A caption" width="50%" />

Notes:

We can visualize these 2 dimensions in a 2D graph.

---

### Dimensions

``` python
grades_df = pd.read_csv("data/quiz2-grade-toy-classification.csv")
grades_df.head()
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
X = grades_df.drop(columns=['quiz2'])
X.shape[1]
```

```out
7
```

Notes:

Recall the quiz dataset that we’ve seem a few times?

How many dimensions (features) would this dataset have?

The number of features in the grades dataset can be checked using
`.shape`.

---

### Dimensions in ML problems

Dimensions:

  - Dimensions≈20 : Low dimensional
  - Dimensions≈1000: Medium dimensional
  - Dimensions≈100,000: High dimensional

Notes:

In ML, usually we deal with high dimensional problems where examples are
hard to visualize.

  - Dimensions≈20 is considered low dimensional
  - Dimensions≈1000 is considered medium dimensional
  - Dimensions≈100,000 is considered high dimensional

---

### Feature vectors

**Feature vector**: a vector composed of feature values associated with
an example.

``` python
train_df.head()
```

```out
     longitude  latitude country
160   -76.4813   44.2307  Canada
127   -81.2496   42.9837  Canada
169   -66.0580   45.2788  Canada
188   -73.2533   45.3057  Canada
187   -67.9245   47.1652  Canada
```

An example feature vector from the cities dataset:

``` python
train_df.drop(columns=["country"]).iloc[0].round(2).to_numpy()
```

```out
array([-76.48,  44.23])
```

An example feature vector from the grading dataset:

``` python
grades_df.drop(columns=['quiz2']).iloc[0].round(2).to_numpy()
```

```out
array([ 1,  1, 92, 93, 84, 91, 92])
```

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

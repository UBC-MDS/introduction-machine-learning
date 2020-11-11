---
type: slides
---

# Support Vector Machines (SVMs) with RBF kernel

Notes: <br>

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

``` python
X_train, y_train = train_df.drop(columns=['country']), train_df['country']
X_test, y_test = test_df.drop(columns=['country']), test_df['country']
```

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

Notes:

Another popular similarity-based algorithm is Support Vector Machines
(SVM).

SVMs use a different similarity metric which is called a “kernel” in SVM
land.

We are going to concentrate on the specific kernel called Radial Basis
Functions (RBFs).

Let’s bring back our trusty cities dataset again before introducing it.

---

``` python
cities_plot = alt.Chart(train_df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue'])))
cities_plot
```
<img src="/module4/cities_plot.png" alt="A caption" width="60%" />

Notes:

---

``` python
from sklearn.svm import SVC
```

``` python
svm = SVC(gamma=0.01)
scores = cross_validate(svm, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002941    0.002157    0.823529     0.842105
1  0.002589    0.001599    0.823529     0.842105
2  0.002295    0.001790    0.727273     0.858209
3  0.002511    0.001731    0.787879     0.843284
4  0.002653    0.001537    0.939394     0.805970
```

``` python
svm_cv_score = scores['test_score'].mean()
svm_cv_score
```

```out
0.8203208556149733
```

Notes:

We must first import the necessary library and then we can get to model
building in the same fashion as we did before.

Here we are importing the `SVC` tool from the `sklearn.svm` library.

For now, just ignore the gamma input argument in `SVC()` we will get to
that soon.

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
X_train, y_train = train_df.drop(columns=['country']), train_df['country']
X_test, y_test = test_df.drop(columns=['country']), test_df['country']
```

```out
SVC(gamma=0.01)
```

<img src="/module4/module4_24/unnamed-chunk-11-1.png" width="100%" />

Notes:

if we plot the decision boundaries of the SVM model and compare it to
K-Nearest Neighbours we can see some similarities.

You can think of SVM with RBF kernel as “smooth KNN”.

---

### SVMs

``` python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train);
scores = cross_validate(knn, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002019    0.003087    0.852941     0.849624
1  0.002339    0.003306    0.764706     0.834586
2  0.002121    0.003092    0.727273     0.850746
3  0.001995    0.002880    0.787879     0.858209
4  0.002351    0.003285    0.878788     0.813433
```

``` python
knn_cv_score = scores['test_score'].mean().round(3)
knn_cv_score
```

```out
0.802
```

``` python
svm_cv_score.round(3)
```

```out
0.82
```

Notes:

Superficially, SVMs are more like weighted 𝑘-NNs.

A test example is positive if on average it looks more like positive
examples than negative examples.

The primary difference between 𝑘 -NNs and SVMs is that - Unlike 𝑘 -NNs,
SVMs only remember the key examples (support vectors). So it’s more
efficient than 𝑘 -NN.

If we compare the scores from the 𝑘-NN model using `n_neighbors=5` and
the scores from the SVM model we get similar results.

---

## SVM Regressor

``` python
from sklearn.svm import SVR
```

Notes:

It should come as no surprise that we can use SVM models for regression
problems as well.

We need to make sure to import SVR from the SVM sklearn library.

---

<br> <br>

### Hyperparameters of SVM are:

    - `gamma`
    - `C`

Notes:

We are not going to go into detail about the meaning behind these
parameters but you should be able to explain how they affect training
and testing scores (The fundamental trade-off).

If you wish to learn more on these you can reference
<a href="https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html" target="_blank">Scikit-learn\`’s
explanation of RBF SVM parameters</a>.

---

### Relation of gamma and the fundamental trade-off

`gamma` controls the complexity of a model, just like other
hyperparameters we’ve seen.

  - As gamma ↑, complexity ↑
  - As gamma ↓, complexity ↓

<br> <br> <br>

<img src="/module4/module4_24/unnamed-chunk-16-1.png" width="100%" />

Notes:

---

### Relation of C and the fundamental trade-off

C also affects the fundamental tradeoff.

  - As C ↑, complexity ↑
  - As C ↓, complexity ↓

<br> <br> <br>

<img src="/module4/module4_24/unnamed-chunk-17-1.png" width="100%" />

Notes:

Obtaining opimal validation scores require a hyperparameter search
between both of these to balance the fundamental tradeoff.

We will learn how to search over multiple hyperparameters at a time in
the next module.

---

# Let’s apply what we learned\!

Notes: <br>

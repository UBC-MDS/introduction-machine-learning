---
type: slides
---

# Choosing K (n\_neighbors)

Notes: <br>

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
X = cities_df.drop(columns=["country"])
y = cities_df["country"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=123)
```

``` python
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train.to_numpy());
```

``` python
model.score(X_train,y_train)
```

```out
1.0
```

Notes:

In the last section we saw we could build a `KNeighborsClassifier` in a
similar way to how we’ve built other models.

The primary hyperparameter of the model is `n_neighbors` ( 𝑘 ) which
decides how many neighbours should vote during prediction?

What happens when we play around with `n_neighbors`?

Are we more likely to overfit with a low `n_neighbors` or a high
`n_neighbors`?

Let’s examine the effect of the hyperparameter on our cities data.

---

``` python
k = 1
knn1 = KNeighborsClassifier(n_neighbors=k)
scores = cross_validate(knn1, X_train, y_train, return_train_score = True)
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002412    0.003959    0.710526          1.0
1  0.002313    0.003195    0.684211          1.0
2  0.002552    0.003055    0.842105          1.0
3  0.002301    0.004043    0.702703          1.0
4  0.002231    0.003038    0.837838          1.0
```

``` python
k = 100
knn100 = KNeighborsClassifier(n_neighbors=k)
scores = cross_validate(knn100, X_train, y_train, return_train_score = True)
pd.DataFrame(scores)
```

```out
   fit_time  score_time  test_score  train_score
0  0.002054    0.005572    0.605263     0.600000
1  0.001988    0.004277    0.605263     0.600000
2  0.002049    0.003675    0.605263     0.600000
3  0.002463    0.003555    0.594595     0.602649
4  0.002556    0.003211    0.594595     0.602649
```

Notes:

---

<img src="/module3/module4_18/unnamed-chunk-7-1.png" width="1536" />

Notes:

---

### How to choose `n_neighbors`?

``` python
results_dict = {"n_neighbors": list(), "mean_train_score": list(), "mean_cv_score": list()}

for k in range(1,50,5):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(knn, X_train, y_train, return_train_score = True)
    results_dict["n_neighbors"].append(k)
    results_dict["mean_cv_score"].append(np.mean(scores["test_score"]))
    results_dict["mean_train_score"].append(np.mean(scores["train_score"]))

results_df = pd.DataFrame(results_dict)
results_df
```

```out
   n_neighbors  mean_train_score  mean_cv_score
0            1          1.000000       0.755477
1            6          0.831135       0.792603
2           11          0.819152       0.802987
3           16          0.801863       0.782219
4           21          0.777934       0.766430
5           26          0.755364       0.723613
6           31          0.743391       0.707681
7           36          0.728777       0.707681
8           41          0.706128       0.681223
9           46          0.694155       0.660171
```

Notes:

`n_neighbors` is a hyperparameter.

We can use hyperparameter optimization to choose `n_neighbors`.

---

<br>

<center>

<img src="/module4/K_plot.png" alt="A caption" width="80%" />

</center>

Notes:

Here we see that when `n_neighbors` is equal to 11, the cross validation
score is the highest.

---

``` python
sorted_results_df = results_df.sort_values("mean_cv_score", ascending = False)
sorted_results_df
```

```out
   n_neighbors  mean_train_score  mean_cv_score
2           11          0.819152       0.802987
1            6          0.831135       0.792603
3           16          0.801863       0.782219
4           21          0.777934       0.766430
0            1          1.000000       0.755477
5           26          0.755364       0.723613
6           31          0.743391       0.707681
7           36          0.728777       0.707681
8           41          0.706128       0.681223
9           46          0.694155       0.660171
```

``` python
best_k = sorted_results_df.iloc[0,0]
best_k
```

```out
11
```

Notes:

We can confirm this when we sort the scores.

---

``` python
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train);
print("Test accuracy:", round(knn.score(X_test, y_test), 3))
```

```out
Test accuracy: 0.905
```

Notes:

Now when we build our model with `n_neighbors=11` we can hope that our
test accuracy will be optimized.

---

## Curse of dimensionality

<br> <br>

### As dimensions ↑, score ↓

  - 𝑘 -NN usually works well when the number of dimensions is small.

Notes:

As we increase the number of dimensions, our success at predicting
decreases. This is called `Curse of dimensionality`.

This affects all learners but it’s especially bad for nearest-neighbour.

𝑘 -NN usually works well when the number of dimensions is small but
things fall apart quickly as the number of dimensions goes up.

If there are many irrelevant attributes, 𝑘 -NN is hopelessly confused
because all of them contribute to finding similarity between examples.

With enough irrelevant attributes the accidental similarity swamps out
meaningful similarity and 𝑘 -NN is no better than random guessing.

---

### Other useful arguments of `KNeighborsClassifier`

<center>

<img src="/module4/knn.png" alt="A caption" width="80%" />

</center>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html" target="_blank">**Attribution**
</a>

Notes:

There are many different arguments to use with `KNeighborsClassifier`,
one of them being `weights`.

`weights` allows us to assign higher weight to the examples which are
closer to the query example.

---

# Let’s apply what we learned\!

Notes: <br>

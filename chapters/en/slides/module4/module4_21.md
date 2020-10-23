---
type: slides
---

# 𝑘 -Nearest Neighbours Regressor

Notes: <br>

---

## Regression with 𝑘 -nearest neighbours ( 𝑘 -NNs)

``` python
np.random.seed(0)
n = 50
X_1 = np.linspace(0,2,n)+np.random.randn(n)*0.01
X = pd.DataFrame(X_1[:,None], columns=['length'])
X.head()
```

```out
     length
0  0.017641
1  0.044818
2  0.091420
3  0.144858
4  0.181941
```

``` python
y = np.random.randn(n,1) + X_1[:,None]*5
y = pd.DataFrame(y, columns=['weight'])
y.head()
```

```out
     weight
0 -0.807264
1  0.610992
2 -0.053705
3 -0.456343
4  0.881522
```

Notes:

We can solve regression problems with 𝑘 -nearest neighbours algorithm as
well.

In KNN regression we take the average of the 𝑘 -nearest neighbours.

Let’s say we have a toy dataset with a single feature and 50 examples.

For this we are going to make up a dataset. Let’s say our feature is the
length of a snake 🐍 and we want to predict the weight of it.

(You do not need to worry about the code here)

---

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

``` python
source = pd.concat([X_train, y_train], axis=1)

scatter = alt.Chart(source, width=500, height=300).mark_point(filled=True, color='green').encode(
    alt.X('length:Q'),
    alt.Y('weight:Q'))

scatter
```
<br>

<center>

<img src="/module4/snakes.png" alt="A caption" width="65%" />

</center>

Notes:

Let’s split up our data so we do not break the golden rule of machine
learning.

Then we can plot our training data.

---

``` python
from sklearn.neighbors import KNeighborsRegressor
```

``` python
knnr = KNeighborsRegressor(n_neighbors=1, weights="uniform")
knnr.fit(X_train,y_train);
```

``` python
predicted = knnr.predict(X_train)
predicted[:5]
```

```out
array([[ 4.44737813],
       [11.41658175],
       [ 1.95805847],
       [ 9.51879111],
       [ 1.39987614]])
```

``` python
knnr.score( X_train, y_train)  
```

```out
1.0
```

Notes:

Let’s first import `KNeighborsRegressor`.

We train as we are used to and predict, this time expecting a numerical
value as as target.

When we score it, we get 100% training score and you’ll see why in the
graph next.

---

<br>

<center>

<img src="/module4/snakes_1.png" alt="A caption" width="80%" />

</center>

Notes:

---

``` python
knnr = KNeighborsRegressor(n_neighbors=10, weights="uniform")
knnr.fit(X_train, y_train);
```

``` python
knnr.score(X_train, y_train)
```

```out
0.9322080850328923
```

<center>

<img src="/module4/snakes_10.png" alt="A caption" width="70%" />

</center>

Notes:

Let’s see what happens when we use 𝑘=10.

Our accuracy decreases on our training data and we can see that our gold
line is not intersecting all the points and is not as close to the
actual value like before.

---

## Using weighted distances

``` python
knnr = KNeighborsRegressor(n_neighbors=10, weights="distance")
knnr.fit(X_train, y_train);
```

``` python
knnr.score(X_train, y_train)
```

```out
1.0
```

<center>

<img src="/module4/snakes_weighted.png" alt="A caption" width="65%" />

</center>

Notes:

---

## Pros and Cons of 𝑘 -Nearest Neighbours

### Pros:

  - Easy to understand, interpret.
  - Simply hyperparameter 𝑘 (`n_neighbors`)controlling the fundamental
    tradeoff.
  - Can learn very complex functions given enough data.
  - Lazy learning: Takes no time to `fit`

<br>

### Cons:

  - Can be potentially be VERY slow during prediction time.
  - Often not that great test accuracy compared to the modern
    approaches.
  - You should scale your features. We’ll be looking into it in the next
    lecture.

Notes:

---

# Let’s apply what we learned\!

Notes: <br>

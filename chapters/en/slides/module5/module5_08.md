---
type: slides
---

# Case Study: Preprocessing with Scaling

Notes: <br>

---

``` python
knn_unscaled = KNeighborsClassifier();
knn_unscaled.fit(X_train, y_train)
print('Train score: ', (knn_unscaled.score(X_train, y_train).round(2)))
print('Test score: ', (knn_unscaled.score(X_test, y_test).round(2)))
```

``` out
KNeighborsClassifier()
Train score:  0.71
Test score:  0.45
```

``` python
knn_scaled = KNeighborsClassifier();
knn_scaled.fit(X_train_scaled, y_train)
print('Train score: ', (knn_scaled.score(X_train_scaled, y_train).round(2)))
print('Test score: ', (knn_scaled.score(X_test_scaled, y_test).round(2)))
```

``` out
KNeighborsClassifier()
Train score:  0.94
Test score:  0.89
```

Notes:

We’ve seen why scaling in important when we were using our basketball
dataset in the first section of this module.

In this section we are going to dive a little deeper into the process
and the transformer option.

---

## Scaling

<center>

<img src="/module5/scaling-data.png"  width = "90%" alt="404 image" />

</center>

<a href="https://amueller.github.io/COMS4995-s19/slides/aml-05-preprocessing/#8" target="_blank">Attribution</a>

Notes:

This problem affects a large number of ML methods.

There are a number of approaches to this problem.

We are going to look into two most popular ones; `MinMaxScaler` and
`StandardScaler`.

---

| Approach        | What it does                     | How to update 𝑋 (but see below\!)                    | sklearn implementation                                                                                                                                                            |
| --------------- | -------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Normalization   | sets range to \[0,1\]            | `X -= np.min(X,axis=0)`<br>`X /= np.max(X,axis=0)`   | <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html" target="_blank">`MinMaxScaler()`</a>                                          |
| Standardization | sets sample mean to 0, s.d. to 1 | `X -= np.mean(X,axis=0)`<br>`X /=  np.std(X,axis=0)` | <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler" target="_blank">`StandardScaler()`</a> |

There are all sorts of articles on this: see
<a href="http://www.dataminingblog.com/standardization-vs-normalization/" target="_blank">here</a>
and
<a href="https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc" target="_blank">here</a>.

Notes:

We are going to look into two most popular ones.

---

``` python
pd.DataFrame(X_train_imp, columns = X_train.columns, index = X_train.index).head()
```

```out
       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_household  bedrooms_per_household  population_per_household
6051     -117.75     34.04                22.0       2948.0           636.0      2600.0       602.0         3.1250             4.897010                1.056478                  4.318937
20113    -119.57     37.94                17.0        346.0           130.0        51.0        20.0         3.4861            17.300000                6.500000                  2.550000
14289    -117.13     32.74                46.0       3355.0           768.0      1457.0       708.0         2.6604             4.738701                1.084746                  2.057910
13665    -117.31     34.02                18.0       1634.0           274.0       899.0       285.0         5.2139             5.733333                0.961404                  3.154386
14471    -117.23     32.88                18.0       5566.0          1465.0      6303.0      1458.0         1.8580             3.817558                1.004801                  4.323045
```

``` python
from sklearn.preprocessing import StandardScaler
```

``` python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
pd.DataFrame(X_train_scaled, columns=X_train.columns).head()
```

```out
   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_household  bedrooms_per_household  population_per_household
0   0.908140 -0.743917           -0.526078     0.143120        0.235339    1.026092    0.266135      -0.389736            -0.210591               -0.083813                  0.126398
1  -0.002057  1.083123           -0.923283    -1.049510       -0.969959   -1.206672   -1.253312      -0.198924             4.726412               11.166631                 -0.050132
2   1.218207 -1.352930            1.380504     0.329670        0.549764    0.024896    0.542873      -0.635239            -0.273606               -0.025391                 -0.099240
3   1.128188 -0.753286           -0.843842    -0.459154       -0.626949   -0.463877   -0.561467       0.714077             0.122307               -0.280310                  0.010183
4   1.168196 -1.287344           -0.843842     1.343085        2.210026    4.269688    2.500924      -1.059242            -0.640266               -0.190617                  0.126808
```

Notes: Let’s bring in our imputated data that we processed in the last
slide deck.

We’ve seen the `StandardScaler()` function earlier but let’s see what
the transformed data looks like.

---

``` python
knn = KNeighborsRegressor()
knn.fit(X_train_imp, y_train);
knn.score(X_train_imp, y_train).round(3)
```

```out
0.509
```

``` python
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train);
knn.score(X_train_scaled, y_train).round(3)
```

```out
0.809
```

Notes:

Now we can compare our training score with scaled and unscaled data and
see a noticable difference between the two\!

---

``` python
from sklearn.preprocessing import MinMaxScaler
```

``` python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
pd.DataFrame(X_train_scaled, columns=X_train.columns).head()
```

```out
   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  rooms_per_household  bedrooms_per_household  population_per_household
0   0.657371  0.159405            0.411765     0.074928        0.098541    0.072788    0.098832       0.181039             0.028717                0.021437                  0.002918
1   0.476096  0.573858            0.313725     0.008749        0.020019    0.001345    0.003124       0.205942             0.116642                0.182806                  0.001495
2   0.719124  0.021254            0.882353     0.085279        0.119025    0.040752    0.116264       0.148998             0.027594                0.022275                  0.001099
3   0.701195  0.157279            0.333333     0.041508        0.042365    0.025113    0.046703       0.325099             0.034645                0.018619                  0.001981
4   0.709163  0.036132            0.333333     0.141513        0.227188    0.176574    0.239599       0.093661             0.021064                0.019905                  0.002922
```

Notes:

Looking at the data after transforming it with `MinMaxScaler()` we see
this time there are no negative values.

---

``` python
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train);
knn.score(X_train_scaled, y_train).round(3)
```

```out
0.807
```

Notes:

Again similar to `StandardScaler` there is a big difference in the KNN
training performance after scaling the data.

But we saw last week that training score doesn’t tell us much.

We should look at the cross-validation score.

Let’s take a look at that in the next section.

---

# Let’s apply what we learned\!

Notes: <br>

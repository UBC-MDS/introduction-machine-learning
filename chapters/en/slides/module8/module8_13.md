---
type: slides
---

# Predicting probabilities

Notes: <br>

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
X_train, y_train = train_df.drop(columns=["country"], axis=1), train_df["country"]
X_test, y_test = test_df.drop(columns=["country"], axis=1), test_df["country"]

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
lr = LogisticRegression()
lr.fit(X_train, y_train);
```

``` python
lr.predict(X_test[:1])
```

```out
array(['Canada'], dtype=object)
```

Notes:

In the last slide deck, we saw that we can make “hard predictions” with
logistic regression using `predict` but logistic regression also can
make something called “soft predictions”.

---

``` python
lr.predict(X_test[:1])
```

```out
array(['Canada'], dtype=object)
```

``` python
lr.predict_proba(X_test[:1])
```

```out
array([[0.87848688, 0.12151312]])
```

Notes:

“Soft predictions” are when instead of predicting a specific class, the
model returns a probability for each class.

We use `predict_proba` instead of `predict` for this.

This now returns an array with a probability of how confident the model
is for each target class.

We can see that the model is 87.8% sure that example 1 is class 0
(“Canada”) and 12.15% confident that example 1 is class 0 (“USA”).

`predict` works by predicting the class with the highest probability.

---

## How is this being done?

For linear regression we used something like this:

<font size="4"><em> predicted(value) = coefficient<sub>feature1</sub> x
feature1 + coefficient<sub>feature2</sub> x feature2 + … + intercept
</em></font>

But this won’t work with probabilities.

#### **Sigmoid function** (optional)

<img src="/module8/module8_13/unnamed-chunk-7-1.png" width="60%" style="display: block; margin: auto;" />

Notes:

Ok so we have this option but what exactly is happening behind the
scenes?

Because probabilities MUST be between the values of 0 and 1 we need a
tool that will convert the raw model’s output into a range between
\[0,1\].

We currently can’t take the model’s raw output since we get values that
are negative or greater than 1.

We need to use something called a **sigmoid function** which “squashes”
the raw model output from any number into the range \[0,1\].

---

``` python
predict_y = lr.predict(X_train)
predict_y[-5:]
```

```out
array(['Canada', 'Canada', 'USA', 'Canada', 'Canada'], dtype=object)
```

``` python
y_proba = lr.predict_proba(X_train)
y_proba[-5:]
```

```out
array([[0.69848481, 0.30151519],
       [0.76970638, 0.23029362],
       [0.05301712, 0.94698288],
       [0.63294488, 0.36705512],
       [0.81540165, 0.18459835]])
```

Notes:

If we now compare `predict` with `predict_proba` we can see how
`predict` made a prediction based on the probabilities.

---

``` python
data_dict = {"y":y_train, 
             "pred y": predict_y.tolist(),
             "probabilities": y_proba.tolist()}
pd.DataFrame(data_dict).tail(10)
```

```out
          y  pred y                               probabilities
96   Canada  Canada    [0.7047596510140418, 0.2952403489859582]
57      USA     USA   [0.03121394423109436, 0.9687860557689056]
123  Canada  Canada    [0.6537036743991862, 0.3462963256008138]
..      ...     ...                                         ...
66      USA     USA  [0.053017116268726405, 0.9469828837312736]
126  Canada  Canada   [0.6329448842395046, 0.36705511576049543]
109  Canada  Canada    [0.8154016516676702, 0.1845983483323298]

[10 rows x 3 columns]
```

Notes:

Let’s take a look and compare them to the actual correct labels.

We can see that the first example was incorrectly predicted as “Canada”
instead of “USA” but we also see that the model was not extremely
confident in this prediction. It was 69.8% confident.

For the rest of this selection, the model corrected predicted each city
but the model was more confident in some than others.

---

<br> <br>

<img src="/module8/module8_13/unnamed-chunk-11-1.png" width="90%" style="display: block; margin: auto;" />

Notes:

When we use `predict`, we get a decision boundary with either blue or
red, a colour for each class.

With probabilities using `predict_proba`, we can see that the model is
less confident the closer the observations are to the decision boundary.

---

``` python
lr_targets = pd.DataFrame({"y":y_train,
                           "pred y": predict_y.tolist(),
                           "probability_canada": y_proba[:,0].tolist()})
lr_targets.head(3)
```

```out
          y  pred y  probability_canada
160  Canada  Canada            0.704607
127  Canada  Canada            0.563017
169  Canada  Canada            0.838968
```

``` python
lr_targets.sort_values(by='probability_canada')
```

```out
          y  pred y  probability_canada
37      USA     USA            0.006547
78      USA     USA            0.007685
34      USA     USA            0.008317
..      ...     ...                 ...
0       USA  Canada            0.932487
165  Canada  Canada            0.951092
1       USA  Canada            0.961902

[167 rows x 3 columns]
```

Notes:

Let’s find some examples where the model is pretty confident in it’s
predictions.

This time, when we make our dataframe, we are only bringing in the
probability of predicting “Canada”. This is because if we are 10 percent
confident a prediction is “Canada”, the model is 90% confident in “USA”.

Here we can see both extremes.

We are 99.345% (1- 0.006547) confident that city 37 is “USA” and 96.19%
confident that city 1 is “Canada”.

The model got the first example right, but the second one, it didn’t.

Let’s plot this and see why.

---

``` python
X_train.loc[[1,37]]
```

```out
    longitude  latitude
1   -134.4197   58.3019
37   -98.4951   29.4246
```

<img src="/module8/module8_13/unnamed-chunk-15-1.png" width="70%" style="display: block; margin: auto;" />

Notes:

Both points are “USA” cities but we can now see why the model was so
confident in both examples.

The “USA” city it got wrong is likely in Alaska but the model doesn’t
know that and predicts more so on how close and on which side it lies to
the decision boundary.

---

``` python
lr_targets = pd.DataFrame({"y":y_train,
                           "pred y": predict_y.tolist(),
                           "prob_difference": (abs(y_proba[:,0] - y_proba[:,1])).tolist()})
lr_targets.sort_values(by="prob_difference").head()
```

```out
          y pred y  prob_difference
61      USA    USA         0.001719
54      USA    USA         0.020025
13      USA    USA         0.020025
130  Canada    USA         0.022234
92   Canada    USA         0.022234
```

Notes:

Let’s now find an example where the model is less certain on its
prediction.

We can do this by finding the absolute value of the difference between
the two probabilities.

The smaller the value, the more uncertain the model is.

Here we can see that city 61 and 54 have the model pretty stumped.

Let’s plot them and see why.

---

``` python
X_train.loc[[61, 54]]
```

```out
    longitude  latitude
61   -87.9225   43.0350
54   -83.0466   42.3316
```

<img src="/module8/module8_13/unnamed-chunk-18-1.png" width="70%" style="display: block; margin: auto;" />

Notes:

When we plot the cities with the decision boundary, we get a clear
answer.

The cities lie almost completely on the boundary, this makes the model
very divided on how to classify them.

---

# Let’s apply what we learned\!

Notes: <br>

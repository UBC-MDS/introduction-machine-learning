---
type: slides
---

# Coefficients and coef\_

Notes: <br>

---

## Intuition behind linear regression

| listing number | Number of Bedrooms | Number of Bathrooms | Square Footage | Age | Price         |
| -------------- | ------------------ | ------------------- | -------------- | --- | ------------- |
| 1              | 5                  | 6                   | 3000           | 2   | $6.39 million |
| 2              | 1                  | 1                   | 800            | 90  | $1.67 million |
| 3              | 3                  | 2                   | 1875           | 66  | $3.92 million |

Notes:

Unlike with decision trees where we make predictions with rules and
analogy-based models where we predict a certain class using distance to
other examples, linear classifiers use **coefficients** (or sometimes
known as “weights”) associated with features.

---

<center>

<img src="/module8/house_table.png"  width = "50%" alt="404 image" />

</center>

Notes:

We then use these learned coefficients to make predictions.

For example, suppose we are predicting the price of a house and we have
4 features; number of bedrooms, number of bathrooms, square footage, and
age.

---

<center>

<img src="/module8/house_table.png"  width = "35%" alt="404 image" />

</center>

Consider the following listing (example):

| listing number | Number of Bedrooms | Number of Bathrooms | Square Footage | Age |
| -------------- | ------------------ | ------------------- | -------------- | --- |
| 3              | 3                  | 2                   | 1875           | 66  |

<br>

<font size="4"><em> predicted(price) = coefficient<sub>bedrooms</sub> x
\#bedrooms + coefficient<sub>bathrooms</sub> x \#bathrooms +
coefficient<sub>sqfeet</sub> x \#sqfeet + coefficient<sub>age</sub> x
age + intercept </em></font>

<font size="4"><em> predicted(price) = 0.03 x \#bedrooms + 0.04 x
\#bathrooms + 0.002 x \#sqfeet + -0.01 x \#age + intercept </em></font>

<font size="4"><em> predicted(price) = (0.03 x 3) + (0.04 x 2) + (0.002
x 1875) + (-0.01 x 66) + 0 </em></font>

<font size="4"><em> predicted(price) = 3.26 </em></font>

Notes:

---

## Components of a linear model

<br> <br> <br>

<font size="5"><em> predicted(price) =
(<font  color="#b1d78c">coefficient<sub>bedrooms</sub></font> x
<font  color="7bd1ec">\#bedrooms</font>) +
(<font  color="#b1d78c">coefficient<sub>bathrooms</sub></font> x
<font  color="7bd1ec">\#bathrooms</font>) +
(<font  color="#b1d78c">coefficient<sub>sqfeet</sub></font> x
<font  color="7bd1ec">\#sqfeet</font>) +
(<font  color="#b1d78c">coefficient<sub>age</sub></font> x
<font  color="7bd1ec">age</font>) +
<font  color="e8b0d0">intercept</font> </em> </font>

  - <font  color="7bd1ec"> Input features</font>  
  - <font  color="#b1d78c"> Coefficients, one per feature</font>  
  - <font  color="e8b0d0"> Bias or intercept</font>

Notes:

---

``` python
housing_df = pd.read_csv("data/real_estate.csv")
train_df, test_df = train_test_split(housing_df, test_size=0.1, random_state=1)
train_df.head()
```

```out
     house_age  distance_station  num_stores  latitude  longitude  price
172        6.6          90.45606           9  24.97433  121.54310   58.1
230        4.0        2147.37600           3  24.96299  121.51284   33.4
346       13.2        1712.63200           2  24.96412  121.51670   30.8
244        4.8        1559.82700           3  24.97213  121.51627   21.7
367       15.0        1828.31900           2  24.96464  121.51531   20.9
```

``` python
X_train, y_train = train_df.drop(columns =['price']), train_df['price']
X_test, y_test = test_df.drop(columns =['price']), test_df['price']
```

Notes:

Let’s now use `Ridge` with our Taiwan housing dataset again from the
last slide deck. Here, we want to predict the house price.

---

``` python
lm = Ridge()
lm.fit(X_train, y_train);
training_score = lm.score(X_train, y_train)
training_score
```

```out
0.5170145681350131
```

``` python
lm.coef_
```

```out
array([-2.43214368e-01, -5.33723544e-03,  1.25878207e+00,  8.92353624e+00, -1.34523313e+00])
```

Notes:

We can make our model as usual and train it, and assess our training
score.

We saw that with linear classifiers we have coefficients associated with
each feature of our model.

How do we get that? We can use `.coef_` to obtain them from our trained
model.

But how are these useful?

---

``` python
ridge_coeffs = lm.coef_
ridge_coeffs
```

```out
array([-2.43214368e-01, -5.33723544e-03,  1.25878207e+00,  8.92353624e+00, -1.34523313e+00])
```

``` python
words_coeffs_df = pd.DataFrame(data=ridge_coeffs, index=X_train.columns, columns=['Coefficients'])
words_coeffs_df
```

```out
                  Coefficients
house_age            -0.243214
distance_station     -0.005337
num_stores            1.258782
latitude              8.923536
longitude            -1.345233
```

Notes:

One of the primary advantages of linear classifiers is their ability to
interpret models using these coefficients.

What do these mean? Let’s try to make some sense of it here.

We have our coefficients but we should see which feature corresponds to
which coefficient.

We can do that by making a dataframe with both values.

We can use these coefficients to interpret our model. They show us how
much each of these features affects our model’s prediction.

For example, if we had a house with 2 stores nearby, our `num_stores`
value is 2. That means that 2 \* 1.26 = 2.52 will contribute to our
predicted price\!

The negative coefficients work in the opposite way, for example, every
unit increase in age of a house will, subtracts 0.244 from the house’s
predicted value.

---

``` python
words_coeffs_df.abs().sort_values(by='Coefficients')
```

```out
                  Coefficients
distance_station      0.005337
house_age             0.243214
num_stores            1.258782
longitude             1.345233
latitude              8.923536
```

Notes:

In linear models, the coefficients tell us how each feature affects the
prediction.

So, looking at the features which have coefficient with bigger
magnitudes might be useful and contribute more to the prediction.

It’s important to be careful here though because this depends on the
scaling of the features. Larger features will have smaller coefficients,
but if we scale our features before we build our model then they are on
a somewhat level playing field\! (Another reason we should be scaling
our features\!)

---

## Interpreting learned coefficients

<br> <br>

In linear models:

  - if the coefficient is +, then ↑ the feature values ↑ the prediction
    value.  
  - if the coefficient is -, then ↑ the feature values ↓ the prediction
    value.  
  - if the coefficient is 0, the feature is not used in making a
    prediction.

Notes:

In linear models:

  - if the coefficient is positive, then increasing the feature values
    increases the prediction value.  
  - if the coefficient is negative, then increasing the feature values
    decreases the prediction value.  
  - if the coefficient is zero, the feature is not used in making a
    prediction

---

## Predicting

``` python
X_train.iloc[0:1]
```

```out
     house_age  distance_station  num_stores  latitude  longitude
172        6.6          90.45606           9  24.97433   121.5431
```

``` python
lm.predict(X_train.iloc[0:1])
```

```out
array([52.35605528])
```

Notes:

Let’s take a look at a single example here.

The values in this are the input features.

We can use `predict()` on our features to get a prediction of 52.36.

---

``` python
words_coeffs_df.T
```

```out
              house_age  distance_station  num_stores  latitude  longitude
Coefficients  -0.243214         -0.005337    1.258782  8.923536  -1.345233
```

``` python
X_train.iloc[0:1]
```

```out
     house_age  distance_station  num_stores  latitude  longitude
172        6.6          90.45606           9  24.97433   121.5431
```

``` python
intercept = lm.intercept_
intercept
```

```out
-16.240516720277654
```

Notes:

Using our coefficients, and the model’s intercept we can calculate the
model’s predictions ourselves as well.

---

<center>

<font size="4"><em> predicted(price) = coefficient<sub>house\_age</sub>
x house\_age + coefficient<sub>distance\_station</sub> x
distance\_station + coefficient<sub>num\_stores</sub> x num\_stores +
coefficient<sub>latitude</sub> x latitude +
coefficient<sub>longitude</sub> x longitude + intercept </em></font>

</center>

``` python
(ridge_coeffs * X_train.iloc[0:1]).sum(axis=1) + intercept 
```

```out
172    52.356055
dtype: float64
```

``` python
lm.predict(X_train.iloc[0:1])
```

```out
array([52.35605528])
```

Notes:

All of these feature values multiplied by the coefficients then adding
the intercept, contribute to our prediction.

When we do this by hand using the model’s coefficients and intercept, we
get the same as if we used `predict`.

---

# Let’s apply what we learned\!

Notes: <br>

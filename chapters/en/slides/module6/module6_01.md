---
type: slides
---

# Categorical variables: ordinal encoding

Notes: <br>

---

## Remember our case study with the California housing dataset?

``` python
train_df.head()
```

```out
       longitude  latitude  housing_median_age  households  median_income  median_house_value ocean_proximity  rooms_per_household  bedrooms_per_household  population_per_household
6051     -117.75     34.04                22.0       602.0         3.1250            113600.0          INLAND             4.897010                1.056478                  4.318937
20113    -119.57     37.94                17.0        20.0         3.4861            137500.0          INLAND            17.300000                6.500000                  2.550000
14289    -117.13     32.74                46.0       708.0         2.6604            170100.0      NEAR OCEAN             4.738701                1.084746                  2.057910
13665    -117.31     34.02                18.0       285.0         5.2139            129300.0          INLAND             5.733333                0.961404                  3.154386
14471    -117.23     32.88                18.0      1458.0         1.8580            205000.0      NEAR OCEAN             3.817558                1.004801                  4.323045
```

``` python
X_train = train_df.drop(columns=["median_house_value"])
y_train = train_df["median_house_value"]

X_test = test_df.drop(columns=["median_house_value"])
y_test = test_df["median_house_value"]
```

Notes:

Remember in module 6, we preprocessed only the numeric variables of our
California housing dataset.

Early on, before we even did imputation, we dropped the categorical
feature `ocean_proximity` feature from the dataframe.

We just discussed how dropping certain columns is not always the best
idea since we could be dropping potentially useful features in this
task.

Categorical variables can be extremely useful in that they require their
own different kind of preprocessing.

Let’s create our `X_train` and and `X_test` again by keeping the
`ocean_proximity` feature in the data this time.

---

``` python
pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor()),
    ]
)
```

``` python
pipe.fit(X_train, X_train)
```

``` out
ValueError: Cannot use median strategy with non-numeric data:
could not convert string to float: 'INLAND'

Detailed traceback: 
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.8/site-packages/sklearn/pipeline.py", line 330, in fit
    Xt = self._fit(X, y, **fit_params_steps)
  File "/usr/local/lib/python3.8/site-packages/sklearn/pipeline.py", line 292, in _fit
    X, fitted_transformer = fit_transform_one_cached(
  File "/usr/local/lib/python3.8/site-packages/joblib/memory.py", line 352, in __call__
    return self.func(*args, **kwargs)
```

Notes:

Let’s first see what happens when we try to apply a 𝑘-NN model on our
data and preprocess it for imputation and scaling.

Oh no. That’s not good. We get a `ValueError` output.

You see, `scikit-learn` only accepts numeric data as an input and it’s
not sure how to handle the `ocean_proximity` feature.

---

<br> <br>

### So what do we do?

  - Drop the column (not recommended)
  - We can transform categorical features to numeric ones so that we can
    use them in the model
  - There are two transformations we can do:
      - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html" target="_blank">Ordinal
        encoding</a>
      - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html" target="_blank">One-hot
        encoding</a>(recommended in most cases)

Notes:

We could drop the column as we did in the previous module and we get
descend enough scores, but this won’t always be the case.

Just like we said, about dropping a column due to missing values, we
don’t want to throw away information that could be useful for helping
our model and prediction.

Or we can give `scikit-learn` what it wants\! We can transform our
categorical features into numeric ones so we can use them in our models.

There are 2 types of ways we are going to talk about doing this:

  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html" target="_blank">Ordinal
    encoding</a> (occasionally recommended)
  - One-hot encoding (OHE -recommended in most cases)

---

## Ordinal encoding

``` python
X_toy
```

```out
     rating
0      Good
1       Bad
2      Good
3      Good
4       Bad
5   Neutral
6      Good
7      Good
8   Neutral
9   Neutral
10  Neutral
11     Good
12      Bad
13     Good
```

``` python
pd.DataFrame(X_toy['rating'].value_counts()).rename(columns={'rating': 'frequency'}).T
```

```out
           Good  Neutral  Bad
frequency     7        4    3
```

Notes:

Let’s take a look at a dummy dataframe to explain how to use ordinal
encoding.

Here we have a categorical column specifying different movie rating.

---

``` python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(dtype=int)
oe.fit(X_toy);
X_toy_ord = oe.transform(X_toy)

X_toy_ord
```

```out
array([[1],
       [0],
       [1],
       [1],
       [0],
       [2],
       [1],
       [1],
       [2],
       [2],
       [2],
       [1],
       [0],
       [1]])
```

Notes:

Here we simply assign an integer to each of our unique categorical
labels.

We can use sklearn’s
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html" target="_blank">`OrdinalEncoder`</a>.

First, we import `OrdinalEncoder` from `sklearn.preprocessing`.

`OrdinalEncoder` is a transformer just like `SimpleImputer` and
`StandardScaler` so we initial our encoder and then we fit and
transform, just like we did with numeric columns.

---

``` python
encoding_view = X_toy.assign(language_enc=X_toy_ord)
encoding_view
```

```out
     rating  language_enc
0      Good             1
1       Bad             0
2      Good             1
3      Good             1
4       Bad             0
5   Neutral             2
6      Good             1
7      Good             1
8   Neutral             2
9   Neutral             2
10  Neutral             2
11     Good             1
12      Bad             0
13     Good             1
```

Notes:

Since `sklearn`’s transformed output is an array, we can add it next to
our original column to see what happened.

In this case, we can see that each rating has been designated an integer
value.

For example, `Good` is represented by an encoded value of 1 and `Bad` a
value of 0. Good should have a higher value right?

---

``` python
ratings_order = ['Bad', 'Neutral', 'Good']
```

``` python
oe = OrdinalEncoder(categories = [ratings_order], dtype=int)
oe.fit(X_toy);
X_toy_ord = oe.transform(X_toy)

X_toy_ord
```

```out
array([[2],
       [0],
       [2],
       [2],
       [0],
       [1],
       [2],
       [2],
       [1],
       [1],
       [1],
       [2],
       [0],
       [2]])
```

Note:

we can change that by specifying the order in the `categories` argument.

---

``` python
encoding_view = X_toy.assign(language_enc=X_toy_ord)
encoding_view
```

```out
     rating  language_enc
0      Good             2
1       Bad             0
2      Good             2
3      Good             2
4       Bad             0
5   Neutral             1
6      Good             2
7      Good             2
8   Neutral             1
9   Neutral             1
10  Neutral             1
11     Good             2
12      Bad             0
13     Good             2
```

Note:

Now our `Good` rating is giving an ordinal value of 2 and our `Bad` is
0.

---

``` python
X_toy
```

```out
      language
0      English
1   Vietnamese
2      English
3     Mandarin
4      English
5      English
6     Mandarin
7      English
8   Vietnamese
9     Mandarin
10      French
11     Spanish
12    Mandarin
13       Hindi
```

``` python
pd.DataFrame(X_toy['language'].value_counts()).rename(columns={'language': 'frequency'}).T
```

```out
           English  Mandarin  Vietnamese  Spanish  French  Hindi
frequency        5         4           2        1       1      1
```

Notes:

Let’s try this again but with now we have a categorical column
specifying different languages.

---

``` python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(dtype=int)
oe.fit(X_toy);
X_toy_ord = oe.transform(X_toy)

encoding_view = X_toy.assign(language_enc=X_toy_ord)
encoding_view
```

```out
      language  language_enc
0      English             0
1   Vietnamese             5
2      English             0
3     Mandarin             3
4      English             0
5      English             0
6     Mandarin             3
7      English             0
8   Vietnamese             5
9     Mandarin             3
10      French             1
11     Spanish             4
12    Mandarin             3
13       Hindi             2
```

Notes:

We assign an integer to each of our unique categorical labels and now
all our languages are encoded.

Does it make sense to have this column ordinally encoded though?

We are saying in this case thar `Vietnamese` is as close to `spanish` as
`Mandarin` is to `Hindi`. Does that quantification make sense? We can’t
really compare these things and that is why we use One-hot encoding
which is what we will look at in the next section.

---

# Let’s apply what we learned\!

Notes: <br>

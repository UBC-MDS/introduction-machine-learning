---
type: slides
---

# Make - pipelines & column transformers

Notes: <br>

---

``` python
train_df, test_df = train_test_split(adult, test_size=0.2, random_state=42)
train_df.head()
```

```out
       age  workclass  fnlwgt     education  education.num      marital.status      occupation   relationship   race     sex  capital.gain  capital.loss  hours.per.week native.country income
5514    26    Private  256263       HS-grad              9       Never-married    Craft-repair  Not-in-family  White    Male             0             0              25  United-States  <=50K
19777   24    Private  170277       HS-grad              9       Never-married   Other-service  Not-in-family  White  Female             0             0              35  United-States  <=50K
10781   36    Private   75826     Bachelors             13            Divorced    Adm-clerical      Unmarried  White  Female             0             0              40  United-States  <=50K
32240   22  State-gov   24395  Some-college             10  Married-civ-spouse    Adm-clerical           Wife  White  Female             0             0              20  United-States  <=50K
9876    31  Local-gov  356689     Bachelors             13  Married-civ-spouse  Prof-specialty        Husband  White    Male             0             0              40  United-States  <=50K
```

Notes:

Remember our adult census data from assignment 5? Well, we are bringing
back a more complete version of it.

---

``` python
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']

X_test = test_df.drop(columns=['income'])
y_test = test_df['income']
```

Notes:

Remember, we are trying to predict if a row is classified with an income
`<=50K` or `>50K`.

---

``` python
numeric_features = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week"
]

categorical_features = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "sex",
    "native.country"
]
```

Notes:

In the last slide deck, we split our features into numeric and
categorical features which we will do again for this data.

We can also add a new type of feature called `passthrough_features`
these features are the ones that are omitted from being used in our
model.

---

``` python
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)] )

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", SVC())])
```

Notes:

We defined transformations on the numeric and categorical features, on a
column transformer, and on a pipeline.

You’ll also notice that we can specify `remainder="passthrough"` in our
pipeline.

This seems great but it seems quite a lot.

Well, luckily there is another method and tool that is helpful in making
our life easier.

It’s call `make_pipeline()`.

---

### *make\_pipeline* syntax

``` python
model_pipeline = Pipeline(
    steps=[
        ("scaling", StandardScaler()),
        ("clf", SVC())])
```

``` python
model_pipeline = make_pipeline(
            StandardScaler(), SVC())
```

``` python
model_pipeline
```

``` out
Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC())])
```

Notes:

`make_pipeline()` is a shorthand for the `Pipeline()` constructor and
does not permit, naming the steps.

Instead, their names will be set to the lowercase of their types
automatically.

---

``` python
from sklearn.pipeline import make_pipeline
```

``` python
numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder()
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipe = make_pipeline(preprocessor, SVC())
```

Notes:

Let’s create our numeric and categoric pipelines for this data using
`make_pipeline` instead of `Pipeline()`.

Look how much less effort our pipeline took!

Our `ColumnTransformer` may still have the same syntax but guess what?!
We have a solution for that too!

---

## *make\_column\_transformer* syntax

``` python
from sklearn.compose import make_column_transformer
```

so instead of this:

``` python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features) ]
)
```

we can do this:

``` python
preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features))
```

Notes:

Just like `make_pipeline()`, we can make our column transformer with
`make_column_transformer()`.

This eliminates the need to designate names for the numeric and
categorical transformations.

---

So our whole thing becomes:

``` python
numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder())
                
preprocessor = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features))
               
pipe = make_pipeline(preprocessor, SVC())
```

``` python
scores = cross_validate(pipe, X_train, y_train, cv=5, return_train_score=True)
```

``` out
ValueError: Found unknown categories ['Holand-Netherlands'] in column 7 during transform

Detailed traceback: 
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.8/site-packages/sklearn/utils/validation.py", line 72, in inner_f
    return f(**kwargs)
  File "/usr/local/lib/python3.8/site-packages/sklearn/model_selection/_validation.py", line 242, in cross_validate
```

Notes:

Looks nice but it looks like we have a problem with this dataset.

What’s going on here??

---

``` out
ValueError: Found unknown categories ['Holand-Netherlands'] in column 7 during transform
```

``` python
X_train["native.country"].value_counts().tail(5)
```

```out
Outlying-US(Guam-USVI-etc)    12
Hungary                       11
Scotland                      10
Honduras                       7
Holand-Netherlands             1
Name: native.country, dtype: int64
```

Notes:

Let’s look at the error message:

`Found unknown categories ['Holand-Netherlands'] in column 6 during transform`.

This is an issue with our `OneHotEncoder` transformation.

There is only one instance of category `Holand-Netherlands`.

During cross-validation, this is getting put into the validation split.

By default, `OneHotEncoder` throws an error because you might want to
know about this.

---

## How do we fix it?

``` python
numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features))

pipe = make_pipeline(preprocessor, SVC())
```

``` python
scores = cross_validate(pipe, X_train, y_train, cv=5, return_train_score=True)
pd.DataFrame(scores).mean()
```

```out
fit_time       10.120301
score_time      1.534249
test_score      0.855459
train_score     0.867974
dtype: float64
```

Notes:

Simplest fix: Pass `handle_unknown="ignore"` argument to
`OneHotEncoder`.

It creates a row with all zeros.

Do you want this behaviour though?

In that case, “Holland” or “Mars” or “Hogwarts” would all be treated the
same.

Are you expecting to get many unknown categories? Do you want to be able
to distinguish between them?

With this approach, all unknown categories will be represented with all
zeros.

---

### Cases where it’s OK to break the golden rule

-   If it’s some fixed number of categories.

``` python
all_countries = adult["native.country"].unique()
all_countries
```

```out
array(['United-States', nan, 'Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada', 'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran', 'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba', 'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic', 'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
       'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland', 'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)'], dtype=object)
```

``` python
ohe_cat = OneHotEncoder(categories=all_countries)
```

Notes:

Are there any cases where it’s OK to break the golden rule?

-   If it’s some fixed number of categories.

For example, if the categories are provinces/territories of Canada, we
know the possible values and we can just specify them.

If we know the categories, this might be a reasonable time to “violate
the Golden Rule” (look at the test set) and just hard-code all the
categories.

This syntax allows you to pre-define the categories.

---

# Let’s apply what we learned!

Notes: <br>

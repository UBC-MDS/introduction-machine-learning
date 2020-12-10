---
type: slides
---

# Logistic regression

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

Notes:

Next, we are going to introduce to you a new model called **logistic
regression**.

It’s very similar to `Ridge` we saw earlier but this one has some key
differences.

For one, we can use it with classification instead of regression
problems.

For that reason, we are going to bring back our cities dataset we saw at
the beginning of this course.

---

## Setting the stage

``` python
from sklearn.dummy import DummyClassifier
dc = DummyClassifier(strategy="prior")

scores= pd.DataFrame(cross_validate(dc, X_train, y_train, return_train_score=True))
scores
```

```out
   fit_time  score_time  test_score  train_score
0  0.000702    0.000473    0.588235     0.601504
1  0.000527    0.000358    0.588235     0.601504
2  0.000583    0.000400    0.606061     0.597015
3  0.000504    0.000336    0.606061     0.597015
4  0.000491    0.000336    0.606061     0.597015
```

Notes:

Although we don’t always do this in the slides, we should always be
building a baseline model before we do any type of meaningful modeling.

Let’s do that before we get straight into it.

Now we can have a better idea of how well our model performs.

---

``` python
from sklearn.linear_model import LogisticRegression
```

``` python
lr = LogisticRegression()
scores = pd.DataFrame(cross_validate(lr, X_train, y_train, return_train_score=True))
scores
```

```out
   fit_time  score_time  test_score  train_score
0  0.013990    0.003045    0.852941     0.827068
1  0.013649    0.002283    0.823529     0.827068
2  0.012972    0.001695    0.696970     0.858209
3  0.011815    0.001804    0.787879     0.843284
4  0.009755    0.001589    0.939394     0.805970
```

Notes:

We import `LogisticRegression` from the `sklearn.linear_model` library
as we did with `Ridge`.

This time we can see that our training and cross-validation scores have
increased from those of our `DummyClassifier`.

---

## Visualizing our model

<img src="/module8/module8_09/unnamed-chunk-6-1.png" width="70%" style="display: block; margin: auto;" />

Notes:

We saw that with SVMs and decision trees that we could visualize our
model with decision boundaries and we can do the same thing with
logistic regression.

Here, we can see we get a line that separates our two target classes.

---

<br> <br> <br> <br>

<center>

<img src="/module8/triple_graph.png"  width = "100%" alt="404 image" />

</center>

Notes:

If we look at some other models that we did this in comparison for you
can understand a bit more on why we call Logistic Regression a “linear
Classifiers”.

Notice a linear decision boundary (a line in our case).

---

# Coefficients

``` python
lr = LogisticRegression()
lr.fit(X_train, y_train); 
```

``` python
print("Model coefficients:", lr.coef_)
print("Model intercept:", lr.intercept_)
```

``` out
Model coefficients: [[-0.04108149 -0.33683126]]
Model intercept: [10.8869838]
```

``` python
data = {'features': X_train.columns, 'coefficients':lr.coef_[0]}
pd.DataFrame(data)
```

```out
    features  coefficients
0  longitude     -0.041081
1   latitude     -0.336831
```

Notes:

Just like we saw for `Ridge`. we can get the equation of that line and
the coefficients of our `latitude` and `longitude` features using
`.coef_`.

In this case, we see that both are negative coefficients.

We also can see that the coefficient of latitude is larger in magnitude
than that of longitude.

This makes a lot of sense because Canada as a country lies above the USA
and so we expect `latitude` values to contribute more to a prediction
than `longitude` which Canada and the `USA` have quite similar values.

---

## Predictions

``` python
lr.classes_
```

```out
array(['Canada', 'USA'], dtype=object)
```

``` python
example = X_test.iloc[0,:]
example.tolist()
```

```out
[-64.8001, 46.098]
```

``` python
(example.tolist() * lr.coef_).sum(axis=1) + lr.intercept_ 
```

```out
array([-1.97817876])
```

``` python
lr.predict([example])
```

```out
array(['Canada'], dtype=object)
```

Notes:

Again, let’s take an example from our test set and calculate the outcome
using our coefficients and intercept.

We get a value of -1.978.

In `Ridge` our prediction would be the calculated result so -1.97, but
for logistic regression, we check the **sign** of the calculation only.

Our threshold is 0.

  - If the result was positive, it predicts one class; if negative, it
    predicts the other.

That means everything negative corresponds to “Canada” and everything
positive predicts a class of “USA”.

If we use `predict`, it gives us the same result as well\!

These are “hard predictions” but we can also use this for something
called “soft predictions” as well.

(That’s in the next slide deck\!)

---

## Hyperparameter: C (A new one)

``` python
scores_dict ={
"C" :10.0**np.arange(-6,2,1),
"train_score" : list(),
"cv_score" : list(),
}
for C in scores_dict['C']:
    lr_model = LogisticRegression(C=C)
    results = cross_validate(lr_model, X_train, y_train, return_train_score=True)
    scores_dict['train_score'].append(results["train_score"].mean())
    scores_dict['cv_score'].append(results["test_score"].mean())
```

``` python
pd.DataFrame(scores_dict)
```

```out
           C  train_score  cv_score
0   0.000001     0.598810  0.598930
1   0.000010     0.598810  0.598930
2   0.000100     0.664707  0.658645
3   0.001000     0.784424  0.790731
4   0.010000     0.827842  0.826203
5   0.100000     0.832320  0.820143
6   1.000000     0.832320  0.820143
7  10.000000     0.832320  0.820143
```

Notes:

At this point, you should be feeling pretty comfortable with
hyperparameters.

We saw that `Ridge` has the hyperparameter `alpha`, well `C`
(annoyingly) has the opposite effect on the fundamental trade-off.

In general, we say smaller `C` leads to a less complex model (whereas
with `Ridge`, lower `alpha` means higher complexity).

Higher values of `C` leads to more overfitting and lower values to less
overfitting.

---

``` python
param_grid = {
    "C": scipy.stats.uniform(0, 100)}

lr = LogisticRegression()
grid_search = RandomizedSearchCV(lr, param_grid, cv=5, return_train_score=True, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train);
```

```out
Fitting 5 folds for each of 10 candidates, totalling 50 fits

[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    2.0s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    2.1s finished
```

``` python
grid_search.best_params_
```

```out
{'C': 66.12113453497285}
```

``` python
grid_search.best_score_
```

```out
0.8201426024955436
```

Notes:

`LogisticRegression`’s default `C` hyperparameter is 1.

Let’s see what kind of value we get if we do `RandomizedGrid`.

---

## Logistic regression with text data

``` python
X = [
    "URGENT!! As a valued network customer you have been selected to receive a £900 prize reward!",
    "Lol you are always so convincing.",
    "Nah I don't think he goes to usf, he lives around here though",
    "URGENT! You have won a 1 week FREE membership in our £100000 prize Jackpot!",
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]

y = ["spam", "non spam", "non spam", "spam", "spam", "non spam"]
```

Notes:

In one of the practice problems and in the assigment, we apply logistic
regression with text data.

We want to give you a bit of background for this.

Let’s bring back our spam dummy data that we looked at in Modulee 6.

---

``` python
vec = CountVectorizer()
X_transformed = vec.fit_transform(X);
bow_df = pd.DataFrame(X_transformed.toarray(), columns=sorted(vec.vocabulary_), index=X)
bow_df
```

```out
                                                    08002986030  100000  11  900  all  always  are  around  as  been  call  callers  callertune  camera  co  colour  convincing  copy  customer  don  entitled  for  free  friends  goes  ...  oru  our  per  press  prize  receive  request  reward  selected  set  so  the  think  though  to  update  \
URGENT!! As a valued network customer you have ...            0       0   0    1    0       0    0       0   1     1     0        0           0       0   0       0           0     0         1    0         0    0     0        0     0  ...    0    0    0      0      1        1        0       1         1    0   0    0      0       0   1       0   
Lol you are always so convincing.                             0       0   0    0    0       1    1       0   0     0     0        0           0       0   0       0           1     0         0    0         0    0     0        0     0  ...    0    0    0      0      0        0        0       0         0    0   1    0      0       0   0       0   
Nah I don't think he goes to usf, he lives arou...            0       0   0    0    0       0    0       1   0     0     0        0           0       0   0       0           0     0         0    1         0    0     0        0     1  ...    0    0    0      0      0        0        0       0         0    0   0    0      1       1   1       0   
URGENT! You have won a 1 week FREE membership i...            0       1   0    0    0       0    0       0   0     0     0        0           0       0   0       0           0     0         0    0         0    0     1        0     0  ...    0    1    0      0      1        0        0       0         0    0   0    0      0       0   0       0   
Had your mobile 11 months or more? U R entitled...            1       0   1    0    0       0    0       0   0     0     1        0           0       1   1       1           0     0         0    0         1    1     2        0     0  ...    0    0    0      0      0        0        0       0         0    0   0    2      0       0   2       2   
As per your request 'Melle Melle (Oru Minnaminu...            0       0   0    0    1       0    0       0   2     1     0        1           2       0   0       0           0     1         0    0         0    1     0        1     0  ...    1    0    1      1      0        0        1       0         0    1   0    0      0       0   1       0   

                                                    urgent  usf  valued  vettam  week  with  won  you  your  
URGENT!! As a valued network customer you have ...       1    0       1       0     0     0    0    1     0  
Lol you are always so convincing.                        0    0       0       0     0     0    0    1     0  
Nah I don't think he goes to usf, he lives arou...       0    1       0       0     0     0    0    0     0  
URGENT! You have won a 1 week FREE membership i...       1    0       0       0     1     0    1    1     0  
Had your mobile 11 months or more? U R entitled...       0    0       0       0     0     1    0    0     1  
As per your request 'Melle Melle (Oru Minnaminu...       0    0       0       1     0     0    0    0     3  

[6 rows x 72 columns]
```

Notes:

`CountVectorizer` transforms our `review` column into multiple columns
each being a word from the `X` vocabulary.

---

``` python
lr_text_model = LogisticRegression()
lr_text_model.fit(X_transformed, y);
```

``` python
pd.DataFrame({'feature': vec.get_feature_names(),
              'coefficient': lr_text_model.coef_[0]})
```

```out
        feature  coefficient
0   08002986030     0.083722
1        100000     0.147288
2            11     0.083722
3           900     0.154797
4           all    -0.077603
..          ...          ...
67         week     0.147288
68         with     0.083722
69          won     0.147288
70          you     0.111733
71         your    -0.149088

[72 rows x 2 columns]
```

Notes:

That means that each word is a feature in our model and therefore when
we apply logistic regression to our feature table, we get a coefficient
for each word\!

This should help you in understanding how the coefficients contribute to
the predictions of each example for both the practice problems and the
assignment.

---

# Let’s apply what we learned\!

Notes: <br>

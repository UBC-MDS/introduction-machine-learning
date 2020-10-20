---
type: slides
---

# The fundamental tradeoff and the golden rule

Notes: <br>

---

## Reminder:

  - **score\_train**: is our training score (or mean train score from
    cross-validation).

<br>

  - **score\_valid** is our validation score (or mean validation score
    from cross-validation).

<br>

  - **score\_test** is our test score.

Notes:

Before going further, let’s just remind ourselves of the different
possible scores

---

## The “fundamental tradeoff” of supervised learning

<br> <br> <br>

### As model complexity ↑, Score\_train ↑ and Score\_train − Score\_valid tend to ↑.

Notes:

If your model is very simple, like `DummyClassifier()`, then you won’t
really learn any “specific patterns” of the training set, but your model
won’t be very good in general.

This is **underfitting**.

If your model is very complex, like a
`DecisionTreeClassifier(max_depth=None)`, then you will learn unreliable
patterns that get every single training example correct, but there will
be a huge gap between training error and validation error.

This is **overfitting**.

---

## How to pick a model that would generalize better?

``` python
df = pd.read_csv("data/canada_usa_cities.csv")
X = df.drop(columns=["country"])
y = df["country"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
```

Notes:

So how do we deal with this?

How do we avoid both underfitting and overfitting?

First, let’s bring in our data again.

We are using our family Canada and US cities data.

---

``` python
results_dict = {"depth": list(), "mean_train_score": list(), "mean_cv_score": list()}

for depth in range(1,20):
    model = DecisionTreeClassifier(max_depth=depth)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
    results_dict["depth"].append(depth)
    results_dict["mean_cv_score"].append(scores["test_score"].mean())
    results_dict["mean_train_score"].append(scores["train_score"].mean())

results_df = pd.DataFrame(results_dict)
```

Notes:

Here is a typical workflow to pick the best hyperparameters with a
systematic search over some possible hyperparameter values.

---

``` python
results_df
```

```out
    depth  mean_train_score  mean_cv_score
0       1          0.834349       0.809926
1       2          0.844989       0.804044
2       3          0.862967       0.804412
3       4          0.906865       0.840074
4       5          0.918848       0.845956
..    ...               ...            ...
14     15          1.000000       0.815074
15     16          1.000000       0.803309
16     17          1.000000       0.803309
17     18          1.000000       0.803309
18     19          1.000000       0.809191

[19 rows x 3 columns]
```

Notes:

---

``` python
source = results_df.melt(id_vars=['depth'] , 
                              value_vars=['mean_train_score', 'mean_cv_score'], 
                              var_name='plot', value_name='score')
```

``` python
chart1 = alt.Chart(source).mark_line().encode(
    alt.X('depth:Q', axis=alt.Axis(title="Tree Depth")),
    alt.Y('score:Q'),
    alt.Color('plot:N', scale=alt.Scale(domain=['mean_train_score', 'mean_cv_score'],
                                           range=['teal', 'gold'])))
chart1
```
<img src="/module3/chart1.png" alt="A caption" width="60%" />

Notes:

So which hyperparameter do we choose?

There are many subtleties here and there is no perfect answer.

A common practice is to pick the model with minimum cross-validation
error.

---

``` python
results_df.sort_values('mean_cv_score', ascending=False).iloc[0]
```

```out
depth               5.000000
mean_train_score    0.918848
mean_cv_score       0.845956
Name: 4, dtype: float64
```

``` python
best_depth = results_df.sort_values('mean_cv_score', ascending=False).iloc[0]['mean_cv_score']
best_depth
```

```out
0.8459558823529412
```

``` python
model = DecisionTreeClassifier(max_depth=best_depth)
model.fit(X_train, y_train);
print("Score on test set: " + str(round(model.score(X_test, y_test), 2)))
```

```out
Score on test set: 0.67
```

Notes:

Let’s pick `depth=5` which is where the mean cross-validation error is
at a minimum.

Let’s now compare this error with the model’s test error.

Is the test error comparable with the cross-validation error?

Do we feel confident that this model would give a similar performance
when deployed?

---

## The Golden Rule

Even though we care the most about test error:  

<center>

<b>THE TEST DATA CANNOT INFLUENCE THE TRAINING PHASE IN ANY WAY.</b>

</center>

<br>

<img src="/module3/gavel.png" alt="A caption" width="70%" />

Notes:

Even though we care the most about test error **THE TEST DATA CANNOT
INFLUENCE THE TRAINING PHASE IN ANY WAY**.

We have to be very careful not to violate it while developing our ML
pipeline.

Even experts end up breaking it sometimes which leads to misleading
results and a lack of generalization on the real data.

---

## Golden rule violation: Example 1

<center>

<img src="/module3/golden_rule_violation.png" alt="A caption" width="52%" />

</center>

<a href="https://www.theregister.com/2019/07/03/nature_study_earthquakes/" target="_blank">**Attribution:
The A register - Katyanna Quach**</a>

Notes:

*… He attempted to reproduce the research, and found a major flaw: there
was some overlap in the data used to both train and test the model.*

---

## Golden rule violation: Example 2

<center>

<img src="/module3/golden_rule_violation_2.png" alt="A caption" width="60%" />

</center>

<a href="https://www.technologyreview.com/2015/06/04/72951/why-and-how-baidu-cheated-an-artificial-intelligence-test/" target="_blank">**Attribution:MIT
Technology Review- Tom Simonite**</a>

Notes:

*… The Challenge rules state that you must only test your code twice a
week, because there’s an element of chance to the results. Baidu has
admitted that it used multiple email accounts to test its code roughly
200 times in just under six months – over four times what the rules
allow.*

---

## How can we avoid violating the golden rule?

<br> <br> <br>

<center>

<img src='/module3/train-test-split.png' alt="A caption" width="100%" />

</center>

Notes:

Recall that when we split data, we put our test set in an imaginary
vault.

---

<br> <br>

### Here is the workflow we’ll generally follow.

  - **Splitting**: Before doing anything, split the data `X` and `y`
    into `X_train`, `X_test`, `y_train`, `y_test` or `train_df` and
    `test_df` using `train_test_split`.  
  - **Select the best model using cross-validation**: Use
    `cross_validate` with `return_train_score = True` so that we can get
    access to training scores in each fold. (If we want to plot train vs
    validation error plots, for instance.)
  - **Scoring on test data**: Finally score on the test data with the
    chosen hyperparameters to examine the generalization performance.

Notes:

Again, there are many subtleties here and we’ll discuss the golden rule
multiple times throughout the course and in the program.

---

# Let’s apply what we learned\!

Notes: <br>

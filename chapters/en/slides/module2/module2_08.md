---
type: slides
---

# Decision trees with continuous features

Notes: <br>

---

``` python
classification_df = pd.read_csv("data/quiz2-grade-toy-classification.csv")
classification_df.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1   quiz2
0              1                 1    92    93    84    91     92      A+
1              1                 0    94    90    80    83     91  not A+
2              0                 0    78    85    83    80     80  not A+
3              0                 1    91    94    92    91     89      A+
4              0                 1    77    83    90    92     85      A+
```

``` python
X = classification_df.drop(["quiz2"], axis=1)
X.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1
0              1                 1    92    93    84    91     92
1              1                 0    94    90    80    83     91
2              0                 0    78    85    83    80     80
3              0                 1    91    94    92    91     89
4              0                 1    77    83    90    92     85
```

Notes:

We saw how the decision tree works when our data is binary but we know
that this won’t always happen.

What happens when our features are continuous like in our quiz2 data?

All our features here have a numerical value. What do we do?

---

<center>

<img src="/module2/module2_08a.png"  width = "80%" alt="404 image" />

</center>

Notes:

---

``` python
X_subset = X[["lab4", "quiz1"]]
X_subset.head()
```

```out
   lab4  quiz1
0    91     92
1    83     91
2    80     80
3    91     89
4    92     85
```

Notes:

For the next example let’s consider a subset of the data with only two
features.

---

## Decision boundaries

``` python
depth = 1
model = DecisionTreeClassifier(max_depth=depth)
model.fit(X_subset, y)
```

<center>

<img src="/module2/module2_08b.png"  width = "50%" alt="404 image" />

</center>

Notes:

What do we do with learned models?

So far we have been using them to predict the class of a new instance.

Another way to think about them is to ask: what sort of test examples
will the model classify as positive, and what sort will it classify as
negative?

Here we can look at this ***decision stump*** which will show us where
the first feature (`lab4`) makes a divide between an `A+` and `Not A+`.

---

<img src="/module2/module2_08/unnamed-chunk-9-1.png" width="672" />

Notes:

We can assume a geometric view of the data. (More on this soon)

Here the red region corresponds to the “not A+” class and the blue
region corresponds to the “A+” class.

There is a line separating the red region and the blue region which is
called the **decision boundary** of the model.

In our current model, this decision boundary is created by asking one
question.

---

### Another example of decision boundaries

``` python
df = pd.read_csv('data/canada_usa_cities.csv')
df.head()
```

```out
   longitude  latitude country
0  -130.0437   55.9773     USA
1  -134.4197   58.3019     USA
2  -123.0780   48.9854     USA
3  -122.7436   48.9881     USA
4  -122.2691   48.9951     USA
```

Notes:

Here is another example of a decision boundary which can help explain
the concept more visually.

We have the latitude and longitude locations of different cities.

We want to predict if they are Canadian or American cities using these
features.

---

``` python
canada = df[df['country'] == 'Canada']
usa = df[df['country'] == 'USA']
```

``` python
chart1 = alt.Chart(df).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40]), axis=alt.Axis(grid=False)),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60]), axis=alt.Axis(grid=False)),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue']))
)
chart1
```
<img src="/module2/chart1.png" alt="A caption" width="50%" />

Notes:

---

## Real boundary between Canada and USA

<center>

<img src="/module2/canada-us-border.jpg" width="70%">

</center>

**Attribution:**
<a href="https://sovereignlimits.com/blog/u-s-canada-border-history-disputes" target="_blank">sovereignlimits.com</a>

Notes:

---

# Let’s apply what we learned\!

Notes: <br>

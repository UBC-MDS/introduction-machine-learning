---
type: slides
---

# Distances

Notes: <br>

---

``` python
cities_df = pd.read_csv("data/canada_usa_cities.csv")
train_df, test_df = train_test_split(cities_df, test_size=0.2, random_state=123)
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

Let’s return to to the cities dataset that we have been working with.

---

``` python
cities_viz = alt.Chart(train_df, width=500, height=300).mark_circle(size=20, opacity=0.6).encode(
    alt.X('longitude:Q', scale=alt.Scale(domain=[-140, -40])),
    alt.Y('latitude:Q', scale=alt.Scale(domain=[20, 60])),
    alt.Color('country:N', scale=alt.Scale(domain=['Canada', 'USA'],
                                           range=['red', 'blue']))
)
cities_viz
```
Notes:

---

``` python
two_cities = cities_df.sample(2, random_state=42).drop(columns=["country"])
two_cities
```

```out
     longitude  latitude
30    -66.9843   44.8607
171   -80.2632   43.1408
```

<img src="/module4/cities_distance.png" alt="A caption" width="70%" />

Notes:

Let’s take 2 points (two feature vectors) from the cities dataset.

The two sampled points are shown as big black circles.

---

### How do we calculate the distance between the two cities?

``` python
two_cities
```

```out
     longitude  latitude
30    -66.9843   44.8607
171   -80.2632   43.1408
```

Subtract the two cities:

``` python
two_cities.iloc[1] - two_cities.iloc[0]
```

```out
longitude   -13.2789
latitude     -1.7199
dtype: float64
```

Square the differences:

``` python
(two_cities.iloc[1] - two_cities.iloc[0])**2
```

```out
longitude    176.329185
latitude       2.958056
dtype: float64
```

Notes:

---

Sum them up:

``` python
((two_cities.iloc[1] - two_cities.iloc[0])**2).sum()
```

```out
179.28724121999983
```

And then take the square root:

``` python
np.sqrt(np.sum((two_cities.iloc[1] - two_cities.iloc[0])**2))
```

```out
13.389818565611703
```

Notes:

---

``` python
np.sqrt(np.sum((two_cities.iloc[1] - two_cities.iloc[0])**2))
```

```out
13.389818565611703
```

``` python
from sklearn.metrics.pairwise import euclidean_distances
```

``` python
euclidean_distances(two_cities)
```

```out
array([[ 0.        , 13.38981857],
       [13.38981857,  0.        ]])
```

Notes:

This is called the ***Euclidean distance***.

We could skip the 4 steps and instead use a toll from the `sklearn`
library.

---

This may look familiar. That’s because when we are in 2 dimensions it
takes on the equation:

<center>

<img src="/module4/pythagoras.gif"  width = "20%" alt="404 image" />

</center>

<br>

<center>

<img src="/module4/pythag.png"  width = "70%" alt="404 image" />

</center>

Notes:

This is Pythagorean Theorem.

---

For vectors in 3-dimensional space, the Euclidean distance then follows
the equation:

<center>

<img src="/module4/eq3.png"  width = "25%" alt="404 image" />

</center>

For vectors in 4-dimensional space:

<center>

<img src="/module4/eq4.png"  width = "35%" alt="404 image" />

</center>

And so on\!

Notes:

---

# Let’s apply what we learned\!

Notes: <br>

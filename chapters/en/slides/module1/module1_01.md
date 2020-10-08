---
type: slides
---

# What is Supervised Machine Learning?

Notes: <br>

---

## Prevalence of Machine Learning (ML)

<center>

<img src='/module1/examples.png'  width = "75%" alt="404 image" />

</center>

Notes:

Machine learning (ML) is used often in our everyday lives.

Some examples include:

  - Voice assistance
  - Google news
  - Recommender Systems
  - Face recognition
  - Auto completion
  - Stock market predictions
  - Character recognition
  - Self-driving cars
  - Cancer diagnosis
  - Drug Discovery
  - AlphaGo

---

## What is Machine Learning?

  - A field of study that gives computers the ability to learn without
    being explicitly programmed.\* <br> – Arthur Samuel (1959)

<center>

<img src="/module1/traditional-programming-vs-ML.png" height="800" width="800">

</center>

Notes:

So what exactly is machine learning?

According to Arthur Samuel, an American pioneer in the field of computer
gaming and artificial intelligence, it is:

*“A field of study that gives computers the ability to learn without
being explicitly programmed.”*

We see it as a different way to think about problem-solving.

---

## Some concrete examples of supervised learning

<br> <br>

## Example 1: Predict whether a patient has a liver disease or not

*In all the the upcoming examples, Don’t worry about the code. Just
focus on the input and output in each example.*

Notes: To introduce the capabilities of machine learning, we are going
to show you a few examples.

The first example is being able to predict whether a patient has a liver
disease or not.

---

``` python
train_df, test_df = train_test_split(df, test_size=4, random_state=16)
train_df.head()
```

```out
     Age  Total_Bilirubin  Direct_Bilirubin  Alkaline_Phosphotase  Alamine_Aminotransferase  Aspartate_Aminotransferase  Total_Protiens  Albumin  Albumin_and_Globulin_Ratio      Target
13    74              1.1               0.4                   214                        22                          30             8.1      4.1                         1.0     Disease
236   22              0.8               0.2                   300                        57                          40             7.9      3.8                         0.9  No Disease
335   13              0.7               0.1                   182                        24                          19             8.9      4.9                         1.2     Disease
234   40              0.9               0.2                   285                        32                          27             7.7      3.5                         0.8     Disease
159   50              1.2               0.4                   282                        36                          32             7.2      3.9                         1.1     Disease
```

Notes: First we obtain our data from our patients, wrangle it as
necessary and split it up.

---

``` python
from xgboost import XGBClassifier
X_train = train_df.drop(columns=['Target'], axis=1)
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'], axis=1)
model = XGBClassifier()
model.fit(X_train, y_train)
```

Notes: Next, we build a model and train our model using the labels we
already have.

Ignore this output here. It’s just explaining what’s going on in the
model which we will explain soon.

---

``` python
pred_df = pd.DataFrame(
    {"Predicted label": model.predict(X_test).tolist()}
)
df_concat = pd.concat([X_test.reset_index(drop=True), pred_df], axis=1)
df_concat
```

```out
   Age  Total_Bilirubin  Direct_Bilirubin  Alkaline_Phosphotase  Alamine_Aminotransferase  Aspartate_Aminotransferase  Total_Protiens  Albumin  Albumin_and_Globulin_Ratio Predicted label
0   61              0.7               0.2                   145                        53                          41             5.8      2.7                        0.87         Disease
1   42             11.1               6.1                   214                        60                         186             6.9      2.8                        2.80         Disease
2   22              0.8               0.2                   198                        20                          26             6.8      3.9                        1.30      No Disease
3   72              1.7               0.8                   200                        28                          37             6.2      3.0                        0.93         Disease
```

Notes: Next, we take our model and use it to predict on unseen data.

Here we can see that our model is predicting an outcome of the patients
under the `Predicted label` column.

---

<br> <br> <br>

## Example 2: Predict the label of a given image

Notes: We can also use it for image recognition.

---

## Predict labels with associated probability scores for unseen images

``` python
images = glob.glob("test_images/*.*")
for image in images:
    img = Image.open(image)
    img.load()
    plt.imshow(img)
    plt.show()
    df = classify_image(img)
    print(df.to_string(index=False))
```

<img src="/module1/module1_01/unnamed-chunk-6-1.png" width="300" />

``` out
  Class  Probability
      ox     0.869893
  oxcart     0.065034
  sorrel     0.028593
 gazelle     0.010053
```

Notes: Here, we already have a trained model that has been shown
hundreds of thousands of images.

If we give it images from our own collection, the model attempts to make
a prediction of the contents of the image.

In this case, the model predicts the animal to be an `ox` with a
probability score 86%. That’s not bad.

---

``` python
images = glob.glob("test_images/*.*")
for image in images:
    img = Image.open(image)
    img.load()
    plt.imshow(img)
    plt.show()
    df = classify_image(img)
    print(df.to_string(index=False))
```

<img src="/module1/module1_01/unnamed-chunk-6-2.png" width="300" />

``` out
            Class  Probability
            llama     0.123625
               ox     0.076333
           kelpie     0.071548
 ibex, Capra ibex     0.060569
```

Notes:

In this case, the model is much less confident in identifying the
animal. The model gives the highest probability score that our donkey
image, is a llama but only with a value of `0.12`.

---

<br> <br> <br>

## Example 3: Predict sentiment expressed in a movie review (pos/neg)

*Attribution: The dataset `imdb_master.csv` was obtained from
<a href="https://www.kaggle.com/uttam94/imdb-mastercsv" target="_blank">Kaggle</a>*

Notes: We also use machine learning to predict negative or positive
sentiment expressed in a movie review.

---

``` python
train_df.head()
```

```out
                                                  review label         file
43020  Just caught it at the Toronto International Fi...   pos   3719_9.txt
49131  The movie itself made me want to go and call s...   pos  9219_10.txt
23701  I came across this movie on DVD purely by chan...   pos   8832_9.txt
4182   Having seen Carlo Lizzani's documentary on Luc...   neg   2514_4.txt
38521  I loved this film. I first saw it when I was 2...   pos  1091_10.txt
```

``` out
                                                review label         file
43020  Just caught it at the Toronto International Fi...   pos   3719_9.txt
49131  The movie itself made me want to go and call s...   pos  9219_10.txt
23701  I came across this movie on DVD purely by chan...   pos   8832_9.txt
4182   Having seen Carlo Lizzani's documentary on Luc...   neg   2514_4.txt
38521  I loved this film. I first saw it when I was 2...   pos  1091_10.txt
```

Notes: First we wrangle our data so that we can train our model.

This data contains the review in a column named `review` and a `label`
column which contains values of either `pos` or `neg` for positive or
negative.

---

``` python
X_train, y_train = train_df['review'], train_df['label']
X_test, y_test = test_df['review'], test_df['label']

clf = Pipeline(
    [
        ("vect", CountVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=5000)),
    ]
)
clf.fit(X_train, y_train)
```

Notes: Next, we build our model and train on our existing data.

Again, don’t worry about the code here.

---

``` python
pred_dict = {
    "reviews": X_test[0:4],
    "true_sentiment": y_test[0:4],
    "sentiment_predictions": clf.predict(X_test[0:4]),
}
pred_df = pd.DataFrame(pred_dict)
pred_df.head()
```

``` out
                                                 reviews prediction sentiment_predictions
43020  Just caught it at the Toronto International Fi...        pos                   pos
49131  The movie itself made me want to go and call s...        pos                   pos
23701  I came across this movie on DVD purely by chan...        pos                   pos
4182   Having seen Carlo Lizzani's documentary on Luc...        neg                   neg
```

Notes: Once we have our model trained, we can then predict data we
haven’t seen before using the model we just built.

This we can see that in these 4 observations, the model correctly
predicts each review’s sentiment.

---

<br> <br> <br>

## Example 4: Predict housing prices

*Attribution: The dataset `kc_house_data.csv` was obtained from
<a href="https://www.kaggle.com/harlfoxem/housesalesprediction" target="_blank">Kaggle</a>.*

Notes: Machine learning can also be used to predict housing prices.

---

``` python
df = pd.read_csv("data/kc_house_data.csv")
df.drop(["id", "date"], axis=1, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)
train_df.head()
```

```out
          price  bedrooms  bathrooms  sqft_living  sqft_lot  floors  waterfront  view  condition  grade  sqft_above  sqft_basement  yr_built  yr_renovated  zipcode      lat     long  sqft_living15  sqft_lot15
8583   509000.0         2       1.50         1930      3521     2.0           0     0          3      8        1930              0      1989             0    98007  47.6092 -122.146           1840        3576
19257  675000.0         5       2.75         2570     12906     2.0           0     0          3      8        2570              0      1987             0    98075  47.5814 -122.050           2580       12927
1295   420000.0         3       1.00         1150      5120     1.0           0     0          4      6         800            350      1946             0    98116  47.5588 -122.392           1220        5120
15670  680000.0         8       2.75         2530      4800     2.0           0     0          4      7        1390           1140      1901             0    98112  47.6241 -122.305           1540        4800
3913   357823.0         3       1.50         1240      9196     1.0           0     0          3      8        1240              0      1968             0    98072  47.7562 -122.094           1690       10800
```

Notes: We wrangle our data just as we did before.

These data consist of the characteristics of houses in King County, USA.

---

``` python

X_train = train_df.drop("price", axis=1)
X_train.head()
```

```out
       bedrooms  bathrooms  sqft_living  sqft_lot  floors  waterfront  view  condition  grade  sqft_above  sqft_basement  yr_built  yr_renovated  zipcode      lat     long  sqft_living15  sqft_lot15
8583          2       1.50         1930      3521     2.0           0     0          3      8        1930              0      1989             0    98007  47.6092 -122.146           1840        3576
19257         5       2.75         2570     12906     2.0           0     0          3      8        2570              0      1987             0    98075  47.5814 -122.050           2580       12927
1295          3       1.00         1150      5120     1.0           0     0          4      6         800            350      1946             0    98116  47.5588 -122.392           1220        5120
15670         8       2.75         2530      4800     2.0           0     0          4      7        1390           1140      1901             0    98112  47.6241 -122.305           1540        4800
3913          3       1.50         1240      9196     1.0           0     0          3      8        1240              0      1968             0    98072  47.7562 -122.094           1690       10800
```

``` python
y_train = train_df["price"]
y_train.head()
```

```out
8583     509000.0
19257    675000.0
1295     420000.0
15670    680000.0
3913     357823.0
Name: price, dtype: float64
```

``` python
X_test = test_df.drop("price", axis=1)
y_test = train_df["price"]
```

Notes:

It’s important that we separate our data from the target column (The `y`
variables).

---

``` python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
```

Notes: We build our model.

---

``` python
pred_df = pd.DataFrame(
    {"Predicted price": model.predict(X_test[0:4]).tolist(), "Actual price": y_test[0:4].tolist()}
)
df_concat = pd.concat([X_test[0:4].reset_index(drop=True), pred_df], axis=1)
df_concat.head()
```

```out
   bedrooms  bathrooms  sqft_living  sqft_lot  floors  waterfront  view  condition  grade  sqft_above  sqft_basement  yr_built  yr_renovated  zipcode      lat     long  sqft_living15  sqft_lot15  Predicted price  Actual price
0         4       2.25         2130      8078     1.0           0     0          4      7        1380            750      1977             0    98055  47.4482 -122.209           2300        8112      333981.6250      509000.0
1         3       2.50         2210      7620     2.0           0     0          3      8        2210              0      1994             0    98052  47.6938 -122.130           1920        7440      615222.4375      675000.0
2         4       1.50         1800      9576     1.0           0     0          4      7        1800              0      1977             0    98045  47.4664 -121.747           1370        9576      329770.0625      420000.0
3         3       2.50         1580      1321     2.0           0     2          3      8        1080            500      2014             0    98107  47.6688 -122.402           1530        1357      565091.6250      680000.0
```

Notes: And we predict on unseen examples using the built model.

If we scroll to right, we can compare the actual price of the house and
the price our model predicted.

---

# Let’s apply what we learned\!

Notes: <br>

---
type: slides
---

# What is Machine Learning

Notes: <br>

---

## Prevalence of ML

Examples:

<center>

<img src='/module1/examples.png'  width = "90%" alt="404 image" />

</center>

Notes:

Machine learning is used often in our everyday lives.

Some examples include:

  - Voice assistance
  - Google news
  - Recommender Systems
  - Face recognition
  - Auto completion
  - Stock market predictions
  - Character recognition
  - Self driving cars
  - Cancer diagnosis
  - Drug Discovery
  - AlphaGo

---

## What is Machine Learning?

> A field of study that gives computers the ability to learn without
> being explicitly programmed. <br> – Arthur Samuel (1959)

<center>

<img src="/module1/traditional-programming-vs-ML.png" height="800" width="800">

</center>

Notes:

ML is a different way to think about problem solving

---

## Some concrete examples

<br> <br>

## Example 1: Predict whether a patient has a liver disease or not

*Do not worry about the code right now. Just focus on the input and
output in each example*

Notes: <br>

---

``` python
df = pd.read_csv("data/indian_liver_patient.csv")
df = df.drop("Gender", axis=1)
df.loc[df["Dataset"] == 1, "Target"] = "Disease"
df.loc[df["Dataset"] == 2, "Target"] = "No Disease"
df = df.drop("Dataset", axis=1)
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

Notes: First we obtain our data and wrangle it as necessary.

---

``` python
from xgboost import XGBClassifier
X_train = train_df.drop(columns=['Target'], axis=1)
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'], axis=1)
y_test = test_df['Target']
model = XGBClassifier()
model.fit(X_train, y_train)
```

```out
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
```

Notes: We are building a model and training our model using the labels
we already have. Ignore this output here. It’s just explaining what’s
going on in the model which we will explain in the upcoming modules.

---

``` python
pred_df = pd.DataFrame(
    {"Predicted label": model.predict(X_test).tolist(), "Actual label": y_test.tolist()}
)
df_concat = pd.concat([X_test.reset_index(drop=True), pred_df], axis=1)
df_concat
```

```out
   Age  Total_Bilirubin  Direct_Bilirubin  Alkaline_Phosphotase  Alamine_Aminotransferase  Aspartate_Aminotransferase  Total_Protiens  Albumin  Albumin_and_Globulin_Ratio Predicted label Actual label
0   61              0.7               0.2                   145                        53                          41             5.8      2.7                        0.87         Disease      Disease
1   42             11.1               6.1                   214                        60                         186             6.9      2.8                        2.80         Disease      Disease
2   22              0.8               0.2                   198                        20                          26             6.8      3.9                        1.30      No Disease      Disease
3   72              1.7               0.8                   200                        28                          37             6.2      3.0                        0.93         Disease      Disease
```

Notes: Next we predict on unseen data using the model we just built.

---

<br> <br> <br>

## Example 2: Predict the label of a given image

Notes: <br>

---

## Predict labels with associated probabilities for unseen images

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

<img src="/module1/module1_01/unnamed-chunk-5-1.png" width="300" />

``` out
  Class  Probability
      ox     0.869893
  oxcart     0.065034
  sorrel     0.028593
 gazelle     0.010053
```

Notes: <br>

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

<img src="/module1/module1_01/unnamed-chunk-5-2.png" width="300" />

``` out
            Class  Probability
            llama     0.123625
               ox     0.076333
           kelpie     0.071548
 ibex, Capra ibex     0.060569
```

Notes: <br>

---

<br> <br> <br>

## Example 3: Predict sentiment expressed in a movie review (pos/neg)

*Attribution: The dataset `imdb_master.csv` was obtained from
<a href="https://www.kaggle.com/uttam94/imdb-mastercsv" target="_blank">Kaggle</a>*

Notes: <br>

---

``` python
imdb_df = pd.read_csv("data/imdb_master.csv", encoding="ISO-8859-1")
imdb_df = imdb_df[imdb_df["label"].str.startswith(("pos", "neg"))]
imdb_df = imdb_df.drop(["Unnamed: 0", "type"], axis=1)
train_df, test_df = train_test_split(imdb_df, test_size=0.10, random_state=12)
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

Notes: First we wrangle our data so that we can train our model

---

``` python
X_train, y_train = train_df['review'], train_df['label']
X_test, y_test = train_df['review'], train_df['label']

clf = Pipeline(
    [
        ("vect", CountVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=5000)),
    ]
)
clf.fit(X_train, y_train)
```

```out
Pipeline(steps=[('vect', CountVectorizer(max_features=5000)),
                ('clf', LogisticRegression(max_iter=5000))])
```

Notes: Next, we build our model

---

``` python
pred_dict = {
    "reviews": X_test[0:4],
    "prediction": y_test[0:4],
    "sentiment_predictions": clf.predict(X_test[0:4]),
}
pred_df = pd.DataFrame(pred_dict)
pred_df.head()
```

```out
                                                 reviews prediction sentiment_predictions
43020  Just caught it at the Toronto International Fi...        pos                   pos
49131  The movie itself made me want to go and call s...        pos                   pos
23701  I came across this movie on DVD purely by chan...        pos                   pos
4182   Having seen Carlo Lizzani's documentary on Luc...        neg                   neg
```

Notes: We then predict on data we haven’t seen before using the model we
just built.

---

<br> <br> <br> \#\# Example 4: Predict housing prices *Attribution: The
dataset `imdb_master.csv` was obtained from
<a href="https://www.kaggle.com/harlfoxem/housesalesprediction" target="_blank">Kaggle</a>*

Notes: <br>

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

---

``` python
import xgboost as xgb
from xgboost import XGBRegressor

X_train, y_train = train_df.drop("price", axis=1), train_df["price"]
X_test, y_test = test_df.drop("price", axis=1), train_df["price"]

model = XGBRegressor()
model.fit(X_train, y_train)
```

```out
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
```

Notes: We build our model.

---

``` python
# Predict on unseen examples using the built model  
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

Notes: And we predict on unseen examples using the built model

---

## Questions to ponder on

  - What are the inputs and outputs in the examples above?
  - How are they different compared to traditional programs, for
    example, calculating factorial of a number?
  - What and how are we exactly “learning” in the above examples? In the
    image classification example, does the model have a concept of cats,
    dogs, and cheetahs?
  - What would it take to predict the correct label for an example the
    algorithm has not seen before?  
  - Are we expected to get correct predictions for all possible
    examples?
  - How do we measure the success or failure of a machine learning
    model? In other words, if you want to use these program in the wild,
    how do you know how reliable it is?  
  - What if the model misclassifies an unseen example? For instance,
    what if the model incorrectly diagnoses a patient with not having
    disease when they actually have the disease? Would it be acceptable?
    What would be the consequences?
  - Is it useful to know more fine-grained predictions (e.g.,
    probabilities as we saw in Example 2) rather than just a yes or a
    no?

Notes: <br>

---

# Let’s apply what we learned\!

Notes: <br>

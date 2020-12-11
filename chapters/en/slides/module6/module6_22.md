---
type: slides
---

# Text data

Notes: <br>

---

<br> <br> <br>

<center>

<img src="/module6/num_xy.png"  width = "50%" alt="404 image" />

</center>

Notes:

Machine Learning algorithms that we have seen so far prefer numeric and
fixed-length input that looks like this.

But what if we are only given data in the form of raw text and
associated labels?

How can we represent such data into a fixed number of features?

---

<br> <br> <br>

<center>

<img src="/module6/cat_xy.png"  width = "100%" alt="404 image" />

</center>

Notes:

Would you be able to apply the algorithms we have seen so far on the
data that looks like this?

In categorical features or ordinal features, we have a fixed number of
categories.

In text features such as above, each feature value (i.e., each text
message) is going to be different.

How do we encode these features?

---

## Bag of words (BOW) representation

<center>

<img src="/module6/bag-of-words.png"  width = "85%" alt="404 image" />

</center>

<a href="https://web.stanford.edu/~jurafsky/slp3/4.pdf" target="_blank">Attribution:
Daniel Jurafsky & James H. Martin</a>

Notes:

One way is to use a simple bag of words (BOW) representation which
involves two components. - The vocabulary (all unique words in all
documents) - A value indicating either the presence or absence or the
count of each word in the document.

---

### Extracting BOW features using *scikit-learn*

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

Let’s say we have 1 feature in our `X` dataframe consisting of the
following text messages.

In our target column, we have the classification of each message as
either `spam` or `non spam`.

---

``` python
from sklearn.feature_extraction.text import CountVectorizer
```

``` python
vec = CountVectorizer()
X_counts = vec.fit_transform(X);
bow_df = pd.DataFrame(X_counts.toarray(), columns=sorted(vec.vocabulary_), index=X)
bow_df
```

```out
                                                    08002986030  100000  11  900  all  always  are  ...  valued  vettam  week  with  won  you  your
URGENT!! As a valued network customer you have ...            0       0   0    1    0       0    0  ...       1       0     0     0    0    1     0
Lol you are always so convincing.                             0       0   0    0    0       1    1  ...       0       0     0     0    0    1     0
Nah I don't think he goes to usf, he lives arou...            0       0   0    0    0       0    0  ...       0       0     0     0    0    0     0
URGENT! You have won a 1 week FREE membership i...            0       1   0    0    0       0    0  ...       0       0     1     0    1    1     0
Had your mobile 11 months or more? U R entitled...            1       0   1    0    0       0    0  ...       0       0     0     1    0    0     1
As per your request 'Melle Melle (Oru Minnaminu...            0       0   0    0    1       0    0  ...       0       1     0     0    0    0     3

[6 rows x 72 columns]
```

Notes:

We import a tool call `CountVectorizer`.

`CountVectorizer` converts a collection of text documents to a matrix of
word counts.

  - Each row represents a “document” (e.g., a text message in our
    example).
  - Each column represents a word in the vocabulary in the training
    data.
  - Each cell represents how often the word occurs in the document.

In the NLP community, a text data set is referred to as a **corpus**
(plural: corpora).

---

### Important hyperparameters of `CountVectorizer`

  - `binary`:
      - Whether to use absence/presence feature values or counts.
  - `max_features`:
      - Only considers top `max_features` ordered by frequency in the
        corpus.
  - `max_df`:
      - When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold.
  - `min_df`:
      - When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold.
  - `ngram_range`:
      - Consider word sequences in the given range.

Notes:

There are many useful and important hyperparameters of
`CountVectorizer`.

---

### Preprocessing

``` python
X
```

``` out
[ "URGENT!! As a valued network customer you have been selected to receive a £900 prize reward!",
  "Lol you are always so convincing.",
  "Nah I don't think he goes to usf, he lives around here though",
  "URGENT! You have won a 1 week FREE membership in our £100000 prize Jackpot!",
  "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
  "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"]
```

``` python
vec.get_feature_names()
```

```out
['08002986030', '100000', '11', '900', 'all', 'always', 'are', 'around', 'as', 'been', 'call', 'callers', 'callertune', 'camera', 'co', 'colour', 'convincing', 'copy', 'customer', 'don', 'entitled', 'for', 'free', 'friends', 'goes', 'had', 'has', 'have', 'he', 'here', 'in', 'jackpot', 'latest', 'lives', 'lol', 'melle', 'membership', 'minnaminunginte', 'mobile', 'mobiles', 'months', 'more', 'nah', 'network', 'nurungu', 'on', 'or', 'oru', 'our', 'per', 'press', 'prize', 'receive', 'request', 'reward', 'selected', 'set', 'so', 'the', 'think', 'though', 'to', 'update', 'urgent', 'usf', 'valued', 'vettam', 'week', 'with', 'won', 'you', 'your']
```

Notes:

`CountVectorizer` is carrying out some preprocessing such as because of
the default argument values.

  - Converting words to lowercase (`lowercase=True`). Take a look at the
    word “urgent” In both cases.
  - getting rid of punctuation and special characters (`token_pattern
    ='(?u)\\b\\w\\w+\\b'`)

---

``` python
param_grid = {"countvectorizer__max_features": range(1,1000)}
```

``` python
pipe = make_pipeline(CountVectorizer(), SVC())

pipe.fit(X, y);
```

``` python
pipe.predict(X)
```

```out
array(['spam', 'non spam', 'non spam', 'spam', 'spam', 'non spam'], dtype='<U8')
```

``` python
pipe.score(X,y)
```

```out
1.0
```

Notes:

We can use `CountVectorizer()` in a pipeline just like any other
transformer.

Here we get a perfect score on the data it’s seen already. How well does
it do on unsceen data?

---

``` python
X_new = [
    "Congratulations! You have been awarded $1000!",
    "Mom, can you pick me up from soccer practice?",
    "I'm trying to bake a cake and I forgot to put sugar in it smh. ",
    "URGENT: please pick up your car at 2pm from servicing",
    "Call 234950323 for a FREE consultation. It's your lucky day!" ]
    
y_new = ["spam", "non spam", "non spam", "non spam", "spam"]
```

``` python
pipe.score(X_new,y_new)
```

```out
0.8
```

Notes:

It’s not perfect but it seems to do well on this data too.

---

### Is this a realistic representation of text data?

Of course, this is not a great representation of language.

  - We are throwing out everything we know about language and losing a
    lot of information.
  - It assumes that there is no syntax and compositional meaning in
    language.

<br> <br> <br>

…But it works surprisingly well for many tasks.

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

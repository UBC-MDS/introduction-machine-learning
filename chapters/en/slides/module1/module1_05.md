---
type: slides
---

# Classification vs Regression

Notes: <br>

---

## Classification vs. Regression

  - **Classification problem**: predicting among two or more discrete
    classes
      - *Example1*: Predict whether a patient has a liver disease or not
      - *Example2*: Predict whether a student would get an A+ or not in
        this course’s quiz2.
  - **Regression problem**: predicting a continuous (typically,
    floating-point) value
      - Example1: Predict housing prices
      - Example2: Predict a student’s score in this course’s quiz2.

Notes:

There are two main kinds of supervised learning problems based on what
they are trying to predict; ***Classification*** and ***Regression***.

**Classification** predicts a target value among two or more discrete
classes.

For example:

  - whether a patient has a liver disease or not
  - whether a student would get an A+ or not in this course’s quiz2.

**Regression** predicts a continuous (typically, floating-point) value.

For example: - housing prices - the scores of students in this course’s
quiz2.

---

<center>

<img src="/module1/classification-vs-regression.png" height="1500" width="1500">

</center>

Notes:

So a classification problem predicts if a student would get an A+ or not
and a regression problem predicts the student’s actual score of the
quiz.

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

Notes: Here is an example of what the data looks like in classification.
Looking at the column `quiz2` which is what we are trying to predict,
the possible values are `A+` or `not A+`.

---

``` python
regression_df = pd.read_csv("data/quiz2-grade-toy-regression.csv")
regression_df.head()
```

```out
   ml_experience  class_attendance  lab1  lab2  lab3  lab4  quiz1  quiz2
0              1                 1    92    93    84    91     92     90
1              1                 0    94    90    80    83     91     84
2              0                 0    78    85    83    80     80     82
3              0                 1    91    94    92    91     89     92
4              0                 1    77    83    90    92     85     90
```

Notes: As for Regression, the column `quiz2` contains numerical values,
like `90` or `82`.

---

# Let’s apply what we learned\!

Notes: <br>

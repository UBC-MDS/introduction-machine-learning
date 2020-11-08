---
type: slides
---

# Train, validation and test split

Notes: <br>

---

<br> <br>

<br>

<center>

<img src="/module3/train-test-split.png"  width = "100%" alt="404 image" />

</center>

Notes:

We’ve talked about how it’s beneficial to the generalization of our
model to split our data into a `train` and a `test` set so we can see
how well the model performs on data it has not seen.

We’ve also talked about hyperparameter tuning.

We do not want to use the test set for this because in that case, it
will no longer truly be “unseen data”.

It’s a good idea to have separate data for tuning the hyperparameters of
a model that is not the test set so that we obtain a model that
generalizes in the best possible way.

This additional data split is called the ***validation*** set.

So we actually want to split our dataset into 3 splits: train,
validation, and test.

We can use the validation data for model tuning (e.g. selecting
hyperparameters) and the test data for a final, “pure” model assessment.

---

### Train/validation/test split

<br> <br>

<center>

<img src="/module3/train-valid-test-split.png"  width = "88%" alt="404 image" />

</center>

<br>

  - **Train**: Used to `fit` our models.
  - **Validation**: Used to assess our model during model tuning.
  - **Test**: Unseen data used for a final assessment.

Notes:

Before going forward, it’s important that you know that there isn’t a
good consensus on the terminology of what is validation and what is test
data.

We will try to use “validation” to refer to data where we have access to
the target values, but unlike the training data, we only use this for
hyperparameter tuning and model assessment; we don’t pass these into
`fit`.

We will try to use “test” to refer to data where we have access to the
target values, but in this case, unlike training and validation data, we
neither use it in training nor hyperparameter optimization.

We only use test data **once** to evaluate the performance of the best
performing model on the validation set.

We lock it in a “vault” until we’re ready to evaluate.

---

## Deployment data

<center>

<img src="/module3/deployment.jpg"  width = "90%" alt="404 image" />

</center>

Notes:

After we build and finalize a model, we deploy it, and then the model
deals with the data in the wild.

We will use “deployment” to refer to this data, where we do **not** have
access to the target values.

Deployment score is the thing we *really* care about.

We use validation and test scores as proxies for deployment score, and
we hope they are similar.

So, if our model does well on the validation and test data, we hope it
will do well on deployment data.

---

|            | `fit` | `score` | `predict` |
| ---------- | ----- | ------- | --------- |
| Train      | ✔️    | ✔️      | ✔️        |
| Validation |       | ✔️      | ✔️        |
| Test       |       | once    | once      |
| Deployment |       |         | ✔️        |

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

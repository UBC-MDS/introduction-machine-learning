---
type: slides
---

# Train, validation and test split

Notes: <br>

---

<br> <br>

<br>

<center>

<img src="/module3/train-valid-test-split.png"  width = "100%" alt="404 image" />

</center>

Notes:

We’ve talked about how it’s beneficial to the generalization of our
model to split our data into a `train` and a `test` set so we can see
how well the model performs on data it has not seen.

We’ve also talked about hyperparameters.

Sometimes it’s a good idea to have a separate data for tuning the
hyperparameters of a model so that we obtain a model that generalizes in
the best possible way.

This data is called the ***validation*** set.

---

### Train/validation/test split

<br> <br>

<center>

<img src="/module3/train-valid-test-split.png"  width = "80%" alt="404 image" />

</center>

<br>

  - **Validation**
  - **Train**
  - **Test**

Notes:

Before going forward, it’s important that you know that there isn’t good
concensus on the terminology of what is validation and what is test.

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

<img src="/module3/deployment.jpg"  width = "100%" alt="404 image" />

</center>

Notes:

After we build and finalize a model, we deploy it, and then the model
deals with the data in the wild.

We will use “deployment” to refer to this data, where we do **not** have
access to the target values.

Deployment error is the thing we *really* care about.

We use validation and test errors as proxies for deployment error, and
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

You can typically expect
\*\*𝐸\_𝑡𝑟𝑎𝑖𝑛\<𝐸\_𝑣𝑎𝑙𝑖𝑑𝑎𝑡𝑖𝑜𝑛\<\_𝐸\_𝑡𝑒𝑠𝑡\<𝐸\_𝑑𝑒𝑝𝑙𝑜𝑦𝑚𝑒𝑛𝑡\*\*.

Notes:

<br>

---

# Let’s apply what we learned\!

Notes: <br>

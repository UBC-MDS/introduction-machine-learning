def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "train_test_split" in __solution__, "Make sure you are calling the train test split function."
    assert "test_size=0.2" in __solution__, "Make sure you are doing 20% for the training and testing split."
    assert "make_pipeline" in __solution__, "Make sure you are creating a pipeline using make_pipeline."
    assert "CountVectorizer()" in __solution__, "Make sure you are specifying CountVectorizer in your pipeline."
    assert "SVC()" in __solution__, "Make sure you are specifying SVC in your pipeline."
    assert "RandomizedSearchCV" in __solution__, "Make sure you are calling RandomizedSearchCV to optimize your hyper parameters."
    assert "fit" in __solution__, "Make sure you are fitting the training data to your model"
    assert tweet_feats > 880, "The best value for max features is incorrect. Are you fitting the model properly?"
    assert round(tweet_val_score,3) > 0.810, "The value for tweet_val_score is incorrect. Are you fitting the model properly?"
    assert round(tweet_test_score,3) > 0.810, "The value for tweet_test_score is incorrect. Are you scoring on the test data?"
    __msg__.good("Nice work, well done!")
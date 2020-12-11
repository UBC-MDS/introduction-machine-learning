def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "make_pipeline" in __solution__, "Make sure creating a pipeline with the preprocessor."
    assert "LogisticRegression" in __solution__, "Make sure you are specifying logistic regression in the pipeline."
    assert 'class_weight="balanced"' in __solution__, "Make sure you are specifying the class weights to be balanced in the pipeline."
    assert "RandomizedSearchCV" in __solution__, "Make sure you using randomized cross validation."
    assert "pmk_search.fit" in __solution__, "Make sure your are fitting the model to the training data."
    assert "n_iter=10" in __solution__, "Make sure you are using the correct number of iterations."
    assert round(pkm_best_c,2) == 33.47, "Your model's value for the best C is incorrect. Are you scoring correctly?"
    assert round(pkm_best_score,2) == 0.67, "Your model's f1 score is incorrect. Are you scoring correctly?"
    assert "pmk_search.predict" in __solution__, "Make sure you are predicting on the test data."
    __msg__.good("Nice work, well done!")
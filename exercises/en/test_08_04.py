def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "make_scorer" in __solution__, "Make sure you are creating a mape scorer"
    assert "" in __solution__, "The value for the mean squared error is incorrect. Are you scoring properly?"
    assert "greater_is_better=False" in __solution__, "Make sure you are specifying that lower numbers are better."
    assert "Ridge()" in __solution__, "Make sure you are building a ridge model."
    assert str(random_search.get_params()['estimator']) == 'Ridge()',  "Make sure you are passing the ridge model to the pipeline"
    assert best_alpha['alpha'] == 0.1, "Your value for best alpha is incorrect. Are you fitting the model correctly?"
    assert round(best_mape,2) == -5.71, "Your value for best mape is incorrect. Are you fitting the model correctly?"
    __msg__.good("Nice work, well done!")
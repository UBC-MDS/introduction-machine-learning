def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "Ridge()" in __solution__, "Make sure you are building a ridge model."
    assert "ridge_bb.fit" in __solution__, "Make sure you are fitting the model on the training data."
    assert round(bb_coeffs[0]) == -3337459.0, "The values for your coefficients are incorrect. Are you fitting the model properly?"
    assert round(bb_coeffs[1]) == 44369.0, "The values for your coefficients are incorrect. Are you fitting the model properly?"
    assert round(bb_intercept) == 11878924.0, "The values for the intercept is incorrect. Are you fitting the model properly?"
    assert round(player_predict[0]) == 9172344.0, "Your predicted values are incorrect. Are you predicting correctly?"
    __msg__.good("Nice work, well done!")
def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "pipe_bb.predict(X_valid)" in __solution__, "Make sure you are predicting on the validation data."
    assert list(predicted_y).count('G') == 33, "Your predicted values are off. Are you predicting on the validation data?"
    assert list(predicted_y).count('F') == 46, "Your predicted values are off. Are you predicting on the validation data?"
    assert "classification_report" in __solution__, "Make sure you are printing the classification report."
    __msg__.good("Nice work, well done!")
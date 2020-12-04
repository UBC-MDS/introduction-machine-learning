def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "pipe_unbalanced.predict(X_valid)" in __solution__, "Make sure you are predictin gon the validation data."
    assert "precision_score" in __solution__, "Make sure you are doing precision scoring on the validation data and the predicted data."
    assert precision > 0.90, "The value for precision is incorrect. Are you scoring properly?"
    assert "recall_score" in __solution__, "Make sure you are doing recall scoring on the validation data and the predicted data."
    assert recall > 0.72, "The value for recall is incorrect. Are you scoring properly?"
    assert "f1_score" in __solution__, "Make sure you are doing f1 scoring on the validation data and the predicted data."
    assert f1 > 0.80, "The value for f1 is incorrect. Are you scoring properly?"
    __msg__.good("Nice work, well done!")
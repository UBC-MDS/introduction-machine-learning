def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "make_pipeline" in __solution__, "Make sure you are creating a pipeline using make_pipeline."
    assert "preprocessor" in __solution__, "Make sure you are including the preprocessor in your pipeline."
    assert "SVC()" in __solution__, "Make sure you are specifying SVC in your pipeline."
    assert "fit" in __solution__, "Make sure you are fitting the training data to your model"
    assert "pipe_unbalanced.predict(X_valid)" in __solution__, "Make sure you are predicting on the validation data."
    assert 'SVC(class_weight="balanced")' in __solution__, "Make sure your specifying to balance the class weights for the balanced pipeline."
    assert "pipe_balanced.fit" in __solution__, "Make sure you are fitting the balanced pipeline on the training data."
    assert "pipe_balanced.predict(X_valid)" in __solution__, "Make sure you are doing prediction for the balanced pipeline on the validation data."
    assert "classification_report" in __solution__, "Make sure you are printing the classification report on the validation and the predicted data."
    assert list(unbalanced_predicted).count(0) == 187, "Your unbalanced predicted values are off. Are you predicting on the validation data?"
    assert list(unbalanced_predicted).count(1) == 5, "Your unbalanced predicted values are off. Are you predicting on the validation data?"
    assert list(balanced_predicted).count(0) == 180, "Your balanced predicted values are off. Are you predicting on the validation data?"
    assert list(balanced_predicted).count(1) == 12, "Your balanced predicted values are off. Are you predicting on the validation data?"
    __msg__.good("Nice work, well done!")
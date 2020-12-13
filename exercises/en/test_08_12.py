def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "import LogisticRegression" in __solution__, "Make sure you are importing the LogisticRegression module."
    assert "make_pipeline" in __solution__, "Make sure creating a pipeline with the preprocessor."
    assert 'logisticregression' in str(list(pkm_pipe.get_params()['steps'])), "Make sure you are specifying logistic regression in the pipeline."
    assert 'class_weight="balanced"' in __solution__, "Make sure you are specifying the class weights to be balanced in the pipeline."
    assert "pkm_pipe.fit" in __solution__, "Make sure your are fitting the model to the training data."
    assert "pkm_pipe.score" in __solution__, "Make sure you are socring the model on the test data."
    assert round(lr_scores,2) == 0.90, "Your model's score is incorrect. Are you scoring correctly?"
    assert [round(x,2) for x in list(pkm_coefs['coefficients'])] == [0.69, 1.13, 1.03, 0.66, 1.12, -0.79], "The pkm_coefs values are incorrect. Are you filling the blanks correctly?"
    __msg__.good("Nice work, well done!")
def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "make_pipeline" in __solution__, "Make sure creating a pipeline with the column transformer provided to you."
    assert "LogisticRegression" in __solution__, "Make sure you are specifying logistic regression in the pipeline."
    assert "lr_pipe.fit" in __solution__, "Make sure your are fitting the model to the training data."
    assert "plot_confusion_matrix" in __solution__, "Make sure you are plotting the confusion matrix of the model."
    __msg__.good("Nice work, well done!")
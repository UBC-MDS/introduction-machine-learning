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
    assert "plot_confusion_matrix" in __solution__, "Make sure you are plotting a confusion matrix on the test data."
    assert "cmap" in __solution__, "Make sure you are setting a color using cmap"
    __msg__.good("Nice work, well done!")
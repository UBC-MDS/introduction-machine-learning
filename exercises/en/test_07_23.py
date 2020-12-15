def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "make_pipeline" in __solution__, "Make sure you are creating a pipeline using make_pipeline."
    assert "preprocessor" in __solution__, "Make sure you are including the preprocessor in your pipeline."
    assert 'SVC(class_weight="balanced")' in __solution__, "Make sure your specifying to balance the class weights for the pipeline."
    assert "cross_validate" in __solution__, "Make sure you are doing cross validation on pipeline."
    assert min(multi_scores['test_accuracy']) > 0.90 and max(multi_scores['test_accuracy']) < 0.99, "The range of your test accuracy is incorrect. Are you fitting the model properly?"
    assert min(multi_scores['train_accuracy']) > 0.96 and max(multi_scores['train_accuracy']) < 0.99, "The range of your training accuracy is incorrect. Are you fitting the model properly?"
    assert min(multi_scores['test_precision']) > 0.49 and max(multi_scores['test_precision']) < 0.85, "The range of your test precision is incorrect. Are you fitting the model properly?"
    assert min(multi_scores['train_precision']) > 0.72 and max(multi_scores['train_precision']) < 0.85, "The range of your training precision is incorrect. Are you fitting the model properly?"
    assert min(multi_scores['test_recall']) > 0.50 and max(multi_scores['test_recall']) == 1, "The range of your test recall is incorrect. Are you fitting the model properly?"
    assert min(multi_scores['train_recall']) == 1 and max(multi_scores['train_recall']) == 1, "The range of your training recall is incorrect. Are you fitting the model properly?"
    __msg__.good("Nice work, well done!")
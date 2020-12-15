def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert sorted(numeric_features) ==  ['age', 'sitting_hrs'], "The numeric features are incorrect. Please try again."
    assert sorted(binary_features) == ['accident_trauma', 'childish_diseases', 'surgical_intervention'], "The binary features are incorrect. Please try again."
    assert sorted(ordinal_features) == ['freq_alcohol_con', 'high_fevers_last_year', 'smoking_habit'], "The ordinal features are incorrect. Please try again."
    assert sorted(categorical_features)  == ['season'], "The categorical features are incorrect. Please try again."
    assert fever_order == ['no', 'more than 3 months ago', 'less than 3 months ago'], "The order for fever is incorrect. Please try again."
    assert smoking_order == [ 'never', 'occasional', 'daily'], "The order for smoking is incorrect. Please try again."
    assert alcohol_order[0:3] == ['hardly ever or never', 'once a week', 'several times a week'], "The order for alcohol is incorrect. Please try again."
    assert "'no', 'more than 3 months ago'" in str(preprocessor.transformers), "Make sure you are specifying all the transformers in the preprocessor."
    assert "'never', 'occasional', 'daily'" in str(preprocessor.transformers), "Make sure you are specifying all the transformers in the preprocessor."
    assert min(pd.DataFrame(scores)['test_score']) > 0.80 and max(pd.DataFrame(scores)['test_score']) < 0.90, "The range of your test scores is incorrect. Are you calling the cross_validate function?"
    assert min(pd.DataFrame(scores)['train_score']) > 0.85 and max(pd.DataFrame(scores)['train_score']) < 0.90, "The range of your training scores is incorrect. Are you calling the cross_validate function?"
    __msg__.good("Nice work, well done!")
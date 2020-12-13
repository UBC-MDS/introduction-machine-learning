def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert 'SimpleImputer(strategy="median")' in __solution__, "Make sure you are using the median strategy for imputing in the numeric transformer."
    assert 'StandardScaler())' in __solution__, "Make sure you are using the standard scaler strategy for scaling in the numeric transformer."
    assert 'SimpleImputer(strategy="most_frequent")' in __solution__, "Make sure you are using the most frequent strategy for imputing in the categorical transformer."
    assert 'OneHotEncoder(handle_unknown="ignore")' in __solution__, "Make sure you are ignoring the unknow cases in the categorical transformer."
    assert '(numeric_transformer, numeric_features)' in __solution__, "Make sure you are including the numeric transformer and numeric features in the column transformer."
    assert '(categorical_transformer, categorical_features)' in __solution__, "Make sure you are including the categorical transformer and numeric features in the column transformer."
    assert 'KNeighborsRegressor()' in __solution__, "Make sure you are specifying KNeighborsRegressor in your main pipeline."
    assert min(pd.DataFrame(with_categorical_scores)['test_score']) > 0.20 and max(pd.DataFrame(with_categorical_scores)['test_score']) < 0.50, "The range of your test scores is incorrect. Are you calling the cross_validate function?"
    assert min(pd.DataFrame(with_categorical_scores)['train_score']) > 0.57 and max(pd.DataFrame(with_categorical_scores)['train_score']) < 0.65, "The range of your training scores is incorrect. Are you calling the cross_validate function?"
    __msg__.good("Nice work, well done!")
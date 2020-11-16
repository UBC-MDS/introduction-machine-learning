def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.

    # If an assertion fails, the message will be displayed
    assert 'SimpleImputer' in __solution__, "Make sure you are using the SimpleImputer() function."
    assert 'median' in __solution__, "Make sure you are using the median strategy for imputation"
    assert "imputer.fit" in __solution__,  "Make sure you are fitting the imputation model on the training data."
    assert __solution__.count('imputer.transform') == 2, "Make sure you are transforming both the training and the test data."
    assert X_train_imp_df.isnull().any(axis=1).sum() == 0, "There is missing data in your dataframe. Are you imputing properly?"
    __msg__.good("Nice work, well done!")

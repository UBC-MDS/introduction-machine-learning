def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert 'OneHotEncoder' in __solution__, "Make sure you are creating a 'OneHotEncoder' object."
    assert 'sparse=False' in __solution__, "Make sure you are setting the sparse argument to false in your OneHotEncoder object."
    assert 'one_hot_encoder.transform' in __solution__, "Make sure you are are transforming the column of interest."
    assert sum(sum(country_encoded)) == 245, "Some of your values in country_encoded are incorrect. Are you specifying the correct column?"
    __msg__.good("Nice work, well done!")


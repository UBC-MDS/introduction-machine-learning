def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert 'OrdinalEncoder(dtype=int)' in __solution__, "Make sure you are specifying an OrdinalEncoder with an integer data type."
    assert "ordinal_encoder.fit" in __solution__, "Make sure you are fitting this model on the column of interest."
    flat_list = [item for sublist in country_encoded for item in sublist]
    assert max(flat_list) == 22, "Make sure you are passing the correct column to the ordinal encoder."
    assert min(flat_list) == 0, "Make sure you are passing the correct column to the ordinal encoder."
    __msg__.good("Nice work, well done!")
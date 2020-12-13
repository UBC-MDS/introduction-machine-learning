def test():
    # Here we can either check objects created in the solution code, or the
    # string value of the solution, available as __solution__. A helper for
    # printing formatted messages is available as __msg__. See the testTemplate
    # in the meta.json for details.
    import pandas as pd
    # If an assertion fails, the message will be displayed
    assert "mean_squared_error" in __solution__, "Make sure you are doing mean squared error scoring on the validation data and the predicted data."
    assert round(mse_calc) == 78343233817724, "The value for the mean squared error is incorrect. Are you scoring properly?"
    assert "np.sqrt" in __solution__, "Make sure you are doing root mean squared error scoring on the validation data and the predicted data."
    assert round(rmse_calc) == 8851171, "The value for root mean squared error is incorrect. Are you taking the square root?"
    assert "r2_score" in __solution__, "Make sure you are doing R-squared scoring on the validation data and the predicted data."
    assert round(r2_calc,2) == -0.15, "The value for R-squared is incorrect. Are you scoring properly?"
    assert round(mape_calc) == 205, "The MAPE score is incorrect. Are you computing the MAPE function correctly?"
    __msg__.good("Nice work, well done!")
---
title: 'Module 8: Linear Models'
description:
  "This module will teach you about different types of linear models. You will learn how this model can be interpreted as well as its advantages and limitations."
prev: /module7
next: /module9
type: chapter
id: 8
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module8/module8_00" shot="0" start="14:3217" end="15:2304">
</slides>
</exercise>


<exercise id="1" title="Introducing Linear Classifiers"  type="slides, video">
<slides source="module8/module8_01" shot="0" start="13:2011" end="14:1221">
</slides>
</exercise>

<exercise id="2" title= "Introducing Linear Classifiers">


**Question 1**    
Which of the following is a well know `Ridge` hyperparameter?

<choice id="1">

<opt text="<code>beta</code>">

What comes before Beta?

</opt>

<opt text= "<code>alpha</code>"  correct="true">
 
Super!

</opt>

<opt text="<code>a</code>">

Closer but not quiet. 

</opt>

<opt text="<code>alpheba</code>" >

Any chance you have seen the musical Wicked? This is a name, not a hyperparameter. 

</opt>

</choice>


**Question 2**    
If I had 2 features and 1 target column, what would I need to visualize our Ridge model?

<choice id="2" >

<opt text="a point">

That's not it.

</opt>

<opt text= "a line">
 
This is the number of false negatives! 

</opt>

<opt text="a plane"  correct="true">

Great! We would need to use a plane which is 2 dimensional, to visualize our Ridge model in a 3-dimensional space. 

</opt>

<opt text="a 3D object" >

This the number of true negatives. 

</opt>

</choice>

</exercise>



<exercise id="3" title="True or False: Ridge">

**True or False?**     
*Ridge is a regression modeling approach.*

<choice id="1" >
<opt text="True"  correct="true">

You got it.

</opt>

<opt text="False">

It's not a classification approach however you can use it with classification (this is outside the scope of this course).

</opt>

</choice>

**True or False**      
*Increasing `alpha` increases model complexity.*

<choice id="2">
<opt text="True">

I think you are thinking of `gamma` or `C` for `SVM`. If we increase `alpha`, we decrease complexity.

</opt>

<opt text="False"  correct="true" >

Nice work! 

</opt>

</choice >

</exercise>

<exercise id="4" title="Using Ridge">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Using our well know basketball dataset, we are going to build a model using the `height` feature and assess if it can help predict a player's` weight`.

Tasks:   
- Import the Ridge function. 
- Create a MAPE scorer from the `mape` function that we provided. Make sure you specify in the scorer that lower numbers are better for MAPE. 
- Build a Ridge model called `ridge_bb`.
- Use `RandomizedSearchCV` to hyperparameter tune `alpha`. Fill the blanks so it uses `ridge_bb` as an estimator and the values from `param_dist`.
- Fit your grid search on the training data.
- What is the best value for `alpha`? Save it in an object named `best_alpha`.
- What is the best MAPE score? Save it in an object named `best_mape`.

<codeblock id="08_04">

- Are you importing the Ridge function?
- Are you making the MAPE scorer with `make_scorer(mape, greater_is_better=False)`?
- Are you filling in the blank for `RandomizedSearchCV` as  `random_search = RandomizedSearchCV(ridge_bb, param_dist, n_iter=20,cv=5, n_jobs=1, random_state=123, scoring=neg_mape_scorer)`?
- Are you fitting with `random_search.fit(X_train, y_train)`?
- Are you finding the best alpha as `random_search.best_params_`? 
- Are you finding the best score with `random_search.best_score_`? 
</codeblock>

</exercise>


<exercise id="5" title="Coefficients and coef_"  type="slides, video">
<slides source="module8/module8_05" shot="3" start="13:2011" end="14:1221">
</slides>
</exercise>


<exercise id="6" title= "Linear Model Coefficient Questions">

Use the following equation to answer the questions below: 

<center><img src="/module8/backpack.svg"  width = "80%" alt="404 image" /></center>

**Question 1**    
What is our intercept value?

<choice id="1">

<opt text="3.02">

This is the laptop coefficient.

</opt>

<opt text= "0.3" >
 
This is the pencil coefficient.

</opt>

<opt text="0.5"  correct="true">

Nailed it! This is the intercept. 

</opt>

<opt text="0" >

Not 0 this time!

</opt>

</choice>


**Question 2**    
If I had 2 laptops 3 pencils in my backpack, what weight would my model predict for my backpack?

<choice id="2" >

<opt text="6.94">

You are missing the intercept value.

</opt>

<opt text= "7.44"  correct="true">
 
Great!

</opt>

<opt text="10.16" >

Are you calculating for 3 laptops and 2 pencils? 

</opt>

<opt text="4.42" >

I think you forgot a laptop!

</opt>

</choice>

</exercise>



<exercise id="7" title="True or False: Coefficients">

**True or False?**     
*With `Ridge`, we learn one weight per training example.*

<choice id="1" >
<opt text="True" >

It's actually one weight per column feature! 

</opt>

<opt text="False"  correct="true">

You got it. It's one weight per feature! 

</opt>

</choice>

**True or False**      
*Coefficients help us interpret our model.*

<choice id="2">
<opt text="True" correct="true">

Great!

</opt>

<opt text="False">

Coefficients actually explain how the model got to a certain prediction. 

</opt>

</choice >

</exercise>

<exercise id="8" title="Interpreting Ridge">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Using the same Ridge model as we obtain last time, let's calculate what our model would predict for the example we have below: 

```
   height  weight
0    2.05    93.2

```

Tasks:    
- Build and fit a `Ridge` model with default hyperparameters and name it `ridge_bb`. 
- What are the coefficients for this model? Save these in an object named `bb_weights`.
- What is the intercept for this model? Save this in an object named `bb_intercept`.
- Using the weights and intercept discovered above, calculate the model's prediction and save the result in `player_predict`.
- Check your answer using `predict`. 

<codeblock id="08_08">

- Are you fitting your model?
- Are you finding the coefficients using `ridge_bb.coef_`?
print(bb_weights)
- Are you using `ridge_bb.intercept_` to find your model's intercept?
- Are you calculating your model's predictions with `bb_intercept + (bb_weights*player_stats).sum(axis=1)`?
- You can check your calculation using `predict` with `ridge_bb.predict(player_stats)`.


</codeblock>

</exercise>

<exercise id="9" title="Logistic Regression"  type="slides, video">
<slides source="module8/module8_09" shot="3" start="13:2011" end="14:1221">
</slides>
</exercise>


<exercise id="10" title= "Logistic Regression Prediction">

Use the following coefficients and intercept to answer the next 2 questions. 


|   Word            | Coefficient | 
|--------------------|-------------|
|excellent           | 2.2         | 
|disappointment      | -2.4        |
|flawless            | 1.4         |
|boring              | -1.3        |
|unwatchable         | -1.7        |
|incoherent          | -1.9        |
|subtle              | 1.3         |

Intercept = 1.3


**Question 1**   

<em>I thought it was going to be excellent but instead, it was unwatchable and boring. </em>

What value do you calculate after using the weights in the model above for the above review? 
The input feature value would be the number of times the word appears in the review (like `CountVectorizer`).


<choice id="1">

<opt text="0.5"  correct="true">

Nice!

</opt>

<opt text= "0.8" >
 
Are you forgetting the intercept?

</opt>

<opt text="-2.1" >

Are you subtracting the intercept?

</opt>

<opt text="-0.5" >

Are you mixing up the sign?

</opt>

</choice>


**Question 2**    
Would the model classify this review as a positive or negative review?

<choice id="2">

<opt text="Positive review" correct="true">

It's a positive value so the model would classify it as a positive review. 

</opt>

<opt text= "Negative review">
 
The model can't see what we see. It's calculating a value based on the weights and the intercept which results in a positive value. 

</opt>

</choice>

</exercise>



<exercise id="11" title="True or False: Logistic Regression">

**True or False?**     
*Increasing logistic regression's `C` hyperparameter increases the model's complexity.*

<choice id="1" >
<opt text="True" correct="true">

You are correct!

</opt>

<opt text="False"  >

Are you mixing this up with `Ridge`'s `alpha` hyperparameter?

</opt>

</choice>

**True or False**      
*Unlike with `Ridge` regression, coefficients are not interpretable with logistic regression.*

<choice id="2">
<opt text="True">

Coefficients explain how a model got to a certain prediction and how much it contributes to a classification. 

</opt>

<opt text="False" correct="true">

Got it!

</opt>

</choice >

</exercise>


<exercise id="12" title=" Applying Logistic Regression">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's give a warm welcome back to our wonderful Pokémon dataset. We want to see how well our model does with logistic regression. Let's try building a simple model with default parameters to start. 


Tasks:   
- Import the logistic regression function. 
- Build and fit a pipeline containing the column transformer and a logistic regression model and use the parameter `class_weight="balanced"`. Name this pipeline `pkm_pipe`.
- Score your model on the test set using the default accuracy measurement. Save this in an object named `lr_scores`.
- Fill in the blanks below to assess the model's feature coefficients. 

<codeblock id="08_12">


- Are you making the MAPE scorer with `make_scorer(mape, greater_is_better=False)`?
- Are you filling in the blank for `RandomizedSearchCV` as  `random_search = RandomizedSearchCV(ridge_bb, param_dist, n_iter=20,cv=5, n_jobs=1, random_state=123, scoring=neg_mape_scorer)`?
- Are you fitting with `random_search.fit(X_train, y_train)`?
- Are you finding the best alpha as `random_search.best_params_`? 
- Are you finding the best score with `random_search.best_score_`? 
- Are you importing the logistic regression function?
- Are you fitting your pipeline?
- Are you scoring your pipeline on the test data?
- Are you finding the coefficients using `pkm_pipe['logisticregression'].coef_[0]`?
- Are you using `numeric_features` to find your model's feature names?

</codeblock>

<br>

**Question 1**    

Which feature contributes the most in predicting if an example is legendary or not? 

<choice id="1">

<opt text="<code>attack</code>">

Which feature has the greatest magnitude?

</opt>

<opt text= "<code>defense</code>"  correct="true">
 
This feature has the highest magnitude!

</opt>

<opt text="<code>sp_attack</code>" >


Which feature has the greatest magnitude?

</opt>

<opt text="<code>sp_defense</code>" >

Which feature has the greatest magnitude?

</opt>

<opt text="<code>speed</code>" >

Which feature has the greatest magnitude?

</opt>

<opt text="<code>capture_rt</code>" >

Which feature has the greatest magnitude?

</opt>

</choice>


**Question 2**    

As the capture rate value increases, will the model more likely predict a legendary or not legendary 
Pokémon?

<choice id="2" >

<opt text="Legendary">

Did you look at the sign of the coefficient?

</opt>

<opt text= "Not Legendary"  correct="true">
 
Great!

</opt>


</choice>


</exercise>

<exercise id="13" title="Predicting Probabilities"  type="slides, video">
<slides source="module8/module8_13" shot="3" start="13:2011" end="14:1221">
</slides>
</exercise>


<exercise id="14" title= "Probabilities and Logistic Regression">

Use the following `.predict_proba()` output to answer the questions below: 

In this case, column 1 is for the classification "hired" and column 2 is "not hired". 

```out
array([[0.04971843, 0.95028157],
       [0.94173513, 0.05826487],
       [0.74133975, 0.25866025],
       [0.13024982, 0.86975018],
       [0.17126403, 0.82873597],
       [0.0483314 , 0.9516686 ],
       [0.21013417, 0.78986583],
       [0.01338452, 0.98661548],
       [0.99508633, 0.00491367],
       [0.99610141, 0.00389859]])
```

**Question 1**    
If we had used `.predict()` for these examples instead of `.predict_proba()`, how many of the examples would the model have predicted "hired"

<choice id="1">

<opt text="10">

This is the total number of examples.

</opt>

<opt text= "6" >
 
This is the number of "not hired" examples.

</opt>

<opt text="4"  correct="true">

Great!

</opt>

<opt text="0" >

Some examples would have been predicted as "hired". How many of these examples have values >0.5 in the first column?  

</opt>

</choice>


**Question 2**    

If the true class labels are below, how many examples would the model have correctly predicted with `predict()`? 

```out
['hired', 'hired', 'hired', 'not hired', 'not hired', 'not hired', 'hired', 'not hired', 'hired', 'hired']
```

<choice id="2" >

<opt text="10">

The model didn't get them all right. Take a closer look.

</opt>

<opt text= "8"  correct="true">
 
Great!

</opt>

<opt text="2" >

This is the number of incorrectly predicted examples.

</opt>

<opt text="0" >

The model did better than 0 right!

</opt>

</choice>

</exercise>



<exercise id="15" title="True or False: predict_proba">

**True or False?**     
*`predict` returns the positive class if the predicted probability of the positive class is greater than 0.5.*

<choice id="1"  correct="true">

<opt text="True" correct="true">

Nice!

</opt>

<opt text="False"  >

Logistic regression's `predict` works by predicting the class which the highest probability (aka; greater than 0.5).

</opt>

</choice>

**True or False**      
*In logistic regression, a function is applied to convert the raw model output into probabilities.*

<choice id="2">
<opt text="True" correct="true">

Great! It's the sigmoid function!

</opt>

<opt text="False">

We need to transform the raw model output so that it lies between the values of 0 and 1 somehow! 

</opt>

</choice >

</exercise>


<exercise id="16" title="Applying predict_proba">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's keep working with the Pokémon dataset. This time let's do a bit more. Let's hyperparameter tune our `C` and see if we can find an example where the model is confident in its prediction.


Tasks:   
- Build and fit a pipeline containing the column transformer and a Logistic Regression model that used the parameter `class_weight="balanced"`. Name this pipeline `pkm_pipe`.
- Perform `RandomizedSearchCV` using the parameters specified in `param_grid`. Use `n_iter` equal to 10, 5 cross-validation folds and return the training score.  Set `random_state=2028` and set your scoring argument to `f1`.  Name this object `pkm_grid`.
- Fit your `pmk_grid` on the training data.
- What is the best `C` value? Save it in an object name `pkm_best_c`.
- What is the best f1 score? Save it in an object named `pkm_best_score`.
- Find the predictions of the test set using `predict`. Save this in an object named `predicted_y`.
- Find the target class probabilities of the test set using `predict_proba`. 
- Save this in an object named `proba_y`.
- Take the dataframe `lr_probs` and sort them in descending order of the model's confidence in predicting legendary Pokémon. Save this in an object named `legend_sorted`. 

<codeblock id="08_16">

- Are you using `make_pipeline(preprocessor, LogisticRegression(class_weight="balanced"))` to build your `pkm_pipe` object?
- In `RandomizedSearchCV` are you calling `pkm_pipe` and `param_grid`?
- Are you specifying `n_iter=10` and `scoring = 'f1'`? 
- Are you fitting `pkm_grid` on your training data?
- Are you using `best_params_` to find the most optimal `C` value?
- Are you using `best_score_` to find the best score?
- For `predicted_y`, are you using `km_grid.predict(X_test)`? 
- For  `proba_y` are you using `pkm_grid.predict_proba(X_test)`?
- Are you sorting `lr_probs` by `prob_legend` and setting `ascending = False`?

</codeblock>

</exercise>

<exercise id="17" title="Multi-class Classification"  type="slides, video">
<slides source="module8/module8_17" shot="3" start="13:2011" end="14:1221">
</slides>
</exercise>



<exercise id="18" title= "Multi-Class Questions">

Use the following coefficient output to answer the questions below: 

```out
              Forward   Guard      Other
weight      -0.031025 -0.193441  0.224466
height       0.227869 -1.358500  1.130631
draft_year  -0.017517  0.010280  0.007237
draft_round  0.250149  0.501243 -0.751392
draft_peak  -0.006979 -0.005453  0.012432

```

**Question 1**    
For which feature does an increased value, push the prediction away from the `Other` class?

<choice id="1">

<opt text="<code>weight</code>">

Try looking at the `Other` column. 

</opt>

<opt text="<code>height</code>">
 
Try looking at the `Other` column. 

</opt>

<opt text="<code>draft_year</code>">

Try looking at the `Other` column. 

</opt>

<opt text="<code>draft_round</code>" correct="true">

Nice!

</opt>

<opt text="<code>draft_peak</code>">

Try looking at the `Other` column. 

</opt>

</choice>


**Question 2**    
If there is an increase in the feature value, for which feature does the classification of `Guard` decrease, and the other 2 features `Forward` and `Other` increase? 

<choice id="2" >

<opt text="<code>weight</code>">

In comparison to the other two features, where is the coefficient negative for `Guard` where the coefficient for `Forward` and `Other` are positive? 

</opt>

<opt text="<code>height</code>"  correct="true">
 
Nice work!

</opt>

<opt text="<code>draft_year</code>">

In comparison to the other two features, where is the coefficient negative for `Guard` where the coefficient for `Forward` and `Other` are positive? 

</opt>

<opt text="<code>draft_round</code>">

In comparison to the other two features, where is the coefficient negative for `Guard` where the coefficient for `Forward` and `Other` are positive? 

</opt>

<opt text="<code>draft_peak</code>">

In comparison to the other two features, where is the coefficient negative for `Guard` where the coefficient for `Forward` and `Other` are positive? 

</opt>

</choice>

</exercise>


<exercise id="19" title="True or False: Coefficients">

**True or False?**     
*Decision trees need special attention for multi-class problems.*

<choice id="1" >
<opt text="True" >

I think you might be confusing decision trees with SVMs and Linear models...

</opt>

<opt text="False"  correct="true">

Cool!

</opt>

</choice>

**True or False**      
*When we plot multi-classification data, there is a dimension for each class.*

<choice id="2">
<opt text="True" >

There is  1 dimension for each feature, not each target class. 

</opt>

<opt text="False" correct="true">

You're doing great!

</opt>

</choice >

</exercise>


<exercise id="20" title="Multi Class Revisited">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Bringing back the Basketball dataset, we are going to take a look at how we assess the predictions from a logistic regression model. 


Tasks:   
- Build and fit a pipeline containing the column transformer and a logistic regression model. Name this pipeline `lr_pipe`.
- Fit your pipeline on the training data.
- Plot a confusion matrix for the test set prediction results and answer the questions below. 

<codeblock id="08_20">

- Are you making your `lr_pipe` pipeline with `make_pipeline(col_transformer,LogisticRegression())`?
- Are you Fitting your pipeline on the training set with `lr_pipe.fit(X_train, y_train)`?
- Are you plotting your confusion matrix with `plot_confusion_matrix(lr_pipe, X_test, y_test,cmap="PuRd");`

</codeblock>


**Question 1**    

Calculate the recall if `Other` is considered the positive label? 
_Remember: Recall = TP/(TP+FN)_


<choice id="1">

<opt text="11/19"  >

You are doing TP/ (TP+FP) 

</opt>

<opt text= "8/19" >
 
This is FP/ (TP+FP) 

</opt>

<opt text="11/74" >

This is TP / (TP + TN + FP + FN)

</opt>

<opt text="11/18" correct="true">

Nice work!

</opt>

</choice>


**Question 2**    

If `F` (forward) is the positive class, how many examples in the dataset are negative (true negative values)? 

</codeblock>

<choice id="2" correct="true">

<opt text="50">

Great!

</opt>

<opt text= "32" >
 
this is the number of examples labeled `G`.

</opt>

<opt text="18"  >

this is the number of examples labeled `Other`.

</opt>

<opt text="25" >

This is the total number of incorrect predictions not negative classes

</opt>

</choice>

</exercise>

<exercise id="21" title="What Did We Just Learn?" type="slides, video">
<slides source="module8/module8_end" shot="0" start="15:2305" end="16:2301">
</slides>

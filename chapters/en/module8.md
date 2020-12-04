---
title: 'Module 8: Linear Models'
description:
  "This module will teach you about different types of linear models specifically, logistic regression. You will learn how this model can be interpreted as well as it's advantages and limitations."
prev: /module7
next: /module9
type: chapter
id: 8
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module8/module8_00" shot="0" start="14:3217" end="15:2304">
</slides>
</exercise>


<exercise id="1" title="Introducing Linear Classifiers: Ridge"  type="slides, video">
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

Any chance you have seen the musical Wicked? This is a name not a hyperparameter. 

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

Great! We would need to use a plane which is 2 dimensional, to visualize our Ridge model in a 3 dimensional space. 

</opt>

<opt text="a 3D object" >

This the number of true negatives. 

</opt>

</choice>

</exercise>



<exercise id="3" title="True or False: Ridge">

**True or False?**     
*Ridge is regression modeling appoach.*

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

Using our well know basketball dataset, we are going to build a model using the `height` feature and assess if it can help predict a players `weight`.

Tasks:   
- Import the Ridge library. 
- Create a mape scorer from the map function that we provided. Make sure you specify in the scorer that lower number are better for MAPE. 
- Build a Ridge model called `ridge_bb`.
- Use `RandomizedSearchCV` to hyperparameter tune `alpha`. Fill the blanks so it uses `ridge_bb` as an estimator and the values from `param_dist`.
- Fit your grid search on the training data.
- What is the best value for `alpha`? Save it in an object named `best_alpha`.
- What is the best MAPE score? Save it in an object named `best_mape`.

<codeblock id="08_04">

- Are you importing the Ridge library?
- Are you making the mape scorer with `make_scorer(mape, greater_is_better=False)`?
- Are you filling in the blank for `RandomizedSearchCV` as  `random_search = RandomizedSearchCV(ridge_bb, param_dist, n_iter=20,cv=5, n_jobs=1, random_state=123, scoring=neg_mape_scorer)`?
- Are you fitting with `random_search.fit(X_train, y_train)`?
- Are you finding the best alpha as `random_search.best_params_`? 
- Are you finding the best score with `random_search.best_score_` 
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
If I had 2 laptops 3 pencils in my backpack, What weight would my model predict for my backpack?

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

Coefficients actual explain how how model got to a certain prediction. 

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
- Using the weights and intercept discovered above, calculate the models's prediction and save the result in `player_predict`.
- Check your answer using predict. 

<codeblock id="08_08">

- Are you fitting your model?
- Are you finding the coefficients using `ridge_bb.coef_`?
print(bb_weights)
- Are you using `ridge_bb.intercept_` to find your model's intercept?
- Are you calculating your model's predictiong with `bb_intercept + (bb_weights*player_stats).sum(axis=1)`?
- You can check your calculation using predict by `ridge_bb.predict(player_stats)`.


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
|dissapointment      | -2.4        |
|flawless            | 1.4         |
|boring              | -1.3        |
|unwatchable         | -1.7        |
|incoherent          | -1.9        |
|subtle              | 1.3         |

Intercept = 1.3


**Question 1**    

What value do you calculate after using the weights in the model above for the following review? The input feature value would be the number of times the word appears in the review (like `CountVectorizer`).

<em>I thought it was going to be excellent but instead it was unwatchable and boring. </em>

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

Are you mixing up the sign.

</opt>

</choice>


**Question 2**    
Would the model classify this review as positive or negative?

<choice id="2">

<opt text="positive" correct="true">

It's a positive value so the model would classify it as a positive review. 

</opt>

<opt text= "negative">
 
The model can't see what we see. It's calculating a value based on the weights and intercept and it results in a positive value. 

</opt>

</choice>

</exercise>



<exercise id="11" title="True or False: Logistic Regression">

**True or False?**     
*Increasing logistic Regression's `C` hyperparameter increases the model's complexity*

<choice id="1" >
<opt text="True" correct="true">

You are correct!

</opt>

<opt text="False"  >

Are you mixing this up with `Ridge`'s `alpha` hyperparameter?

</opt>

</choice>

**True or False**      
*Unlike with `Ridge` regression, coefficients are not interpretable with logistic Regression.*

<choice id="2">
<opt text="True">

Coefficients actual explain how how model got to a certain prediction and much it contribute to a classification. 

</opt>

<opt text="False" correct="true">

Got it!

</opt>

</choice >

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

Great

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
*`predict` returns the positive class if the predicted probability of the positive class is greater than 0.5*

<choice id="1" correct="true">
<opt text="True" >

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

We need to tranform the raw model output so theat they lie between the values of 0 and 1 somehow! 

</opt>

</choice >

</exercise>

<exercise id="17" title="Multi-class Regression"  type="slides, video">
<slides source="module8/module8_17" shot="3" start="13:2011" end="14:1221">
</slides>
</exercise>


<exercise id="18" title= "Multi-Class Questions">

Use the following equation to answer the questions below: 

<center><img src="/module8/backpack.svg"  width = "80%" alt="404 image" /></center>

**Question 1**    
What is our intercept value?

<choice id="1">

<opt text="3.02">

This is the laptop coefficient

</opt>

<opt text= "0.3" >
 
This is the pencil coefficient

</opt>

<opt text="0.5"  correct="true">

Nailed it! This is the intercept. 

</opt>

<opt text="0" >

Not 0 this time!

</opt>

</choice>


**Question 2**    
If I had 2 laptops 3 pencils in my backpack, What weight would my model predict for my backpack?

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



<exercise id="19" title="True or False: Coefficients">

**True or False?**     
*Decision Ttrees need special attention for multi-class problems.*

<choice id="1" >
<opt text="True" >

I think you might be confusing decision trees with SVMs and Linear models...

</opt>

<opt text="False"  correct="true">

Cool!

</opt>

</choice>

**True or False**      
*Coefficients help us interpret our model.*

<choice id="2">
<opt text="True" correct="true">

Great!

</opt>

<opt text="False">

Coefficients actual explain how how model got to a certain prediction. 

</opt>

</choice >

</exercise>

<exercise id="20" title="What Did We Just Learn?" type="slides, video">
<slides source="module8/module8_end" shot="0" start="15:2305" end="16:2301">
</slides>
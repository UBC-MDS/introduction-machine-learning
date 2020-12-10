---
title: 'Module 7: Assessment and Measurements'
description:
  'This module will teach you how to appropriately assess your model. We will teach you how to evaluate and calculate your model using an assortment of different measurements. '
prev: /module6
next: /module8
type: chapter
id: 7
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">
<slides source="module7/module7_00" shot="0" start="13:2011" end="14:1221">
</slides>

</exercise>

<exercise id="1" title="Introducing Evaluation Metrics"  type="slides, video">
<slides source="module7/module7_01" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>

<exercise id="2" title= "Name That Value!">

<center><img src="/module7/Q2.png"  width = "80%" alt="404 image" /></center>


**Question 1**    

How many examples did the model of this matrix correctly label as "Guard"?

<choice id="1">

<opt text="19">

This is the number of examples the model correctly predicted as **Forward**.

</opt>

<opt text= "3">
 
This is the number of examples the model predicted **Guard** when the true label was **Forward**.

</opt>

<opt text="4">

This is the number of examples the model predicted **Forward** when the true label was **Guard**.

</opt>

<opt text="26"  correct="true">

Nice!

</opt>

</choice>


**Question 2**    

If **Forward** is the positive label, how many ***false positive*** values are there?

<choice id="2" >

<opt text="19">

This is the number of examples of true positives. 

</opt>

<opt text= "3">
 
This is the number of false negatives! 

</opt>

<opt text="4"  correct="true">

Great! This is the number of examples the model predicted **Forward** (positive) when the true label was **Guard** (negative).

</opt>

<opt text="26" >

This the number of true negatives. 

</opt>

</choice>




**Question 3**    

How many examples does the model incorrectly predict?

<choice id="3" >

<opt text="45">

This is the number of correctly predicted examples 

</opt>

<opt text= "3">
 
This is the number of false negatives, what about the false positives?

</opt>

<opt text="7"  correct="true">

Great! This is the number of examples the model predicted **Forward** (positive) when the true label was **Guard** (negative).

</opt>

<opt text="4" >

This is the number of false positives, what about the false negatives?

</opt>

</choice>

</exercise>



<exercise id="3" title="True or False: Confusion Matrix">

**True or False?**     
*There are scenarios where accuracy could be misleading..*

<choice id="1" >
<opt text="True" correct="true">

Nice!

</opt>

<opt text="False" >

It's really important to assess your model on not just accuracy.

</opt>

</choice>

**True or False**      
*A confusion matrix will always show the predicted values as columns and the true labels as rows.*

<choice id="2">
<opt text="True">

The matrix axes must be labeled because there is no uniformity to this. 

</opt>

<opt text="False"  correct="true" >

Nice work! 

</opt>

</choice >

</exercise>

<exercise id="4" title="Code a Confusion Matrix">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

We've seen our basketball dataset before and predicted using `SVC` with it before but this time we are going to have a look at how well our model does by building a confusion matrix. 

Tasks:   
- Import the plotting confusion matrix library. 
- Build a pipeline named `pipe_bb` that preprocesses with `preprocessor` and builds an `SVC()` model with default hyperparameters. 
- Fit the pipeline on `X_train` and `y_train`. 
- Next, build a confusion matrix using `plot_confusion_matrix` and calling `pipe_bb` on the **test** set. Pick any colour you like with `cmap`. You can find the colour options <a href=" https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html" target="_blank">here</a>.

<codeblock id="07_04">

- Are you using `make_pipeline(preprocessor, SVC())`?
- Are you fitting your model? 
- Are you calling `pipe_bb`, `X_test` and `y_test` in the `plot_confusion_matrix()` function?

</codeblock>


**Question 1**    
How many players in the test set were correctly predicted as forward?

<choice id="1" >
<opt text="19"  correct="true" >

Nice work!

</opt>

<opt text="2" >

These are wrongly predicted.

</opt>

<opt text="1">

These are wrongly predicted.

</opt>


<opt text="30">

These are correctly predicted as Guards (G). 

</opt>

</choice>


**Question 2**    
If Guard (G) is our positive label, how many false positives are there?

<choice id="2" >
<opt text="19"   >

These are true negatives!

</opt>

<opt text="2" correct="true">

You've picked this up quickly!

</opt>

<opt text="1">

These are false negatives.

</opt>


<opt text="30">

These are true positives!

</opt>

</choice>


</exercise>

<exercise id="5" title="Precision, Recall and F1 Score"  type="slides, video">
<slides source="module7/module7_05" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>


<exercise id="6" title= "Let's Calculate">

<center><img src="/module7/Q2.png"  width = "80%" alt="404 image" /></center>

For the next few questions, use the confusion matrix above and assume that **Forward** is the positive label. 

**Question 1**    

Calculate the recall.

<choice id="1">

<opt text="0.86" correct="true">

Great! 

</opt>

<opt text= "0.83">
 
Are you sure you are calculating recall?

</opt>

<opt text="0.87">

Are you sure you are using Forward are your positive label?

</opt>

<opt text="0.90">

Are you sure you are calculating recall?

</opt>

</choice>


**Question 2**    

Calculate the precision.

<choice id="2" >

<opt text="0.86" >

Are you sure you are calculating recall?

</opt>

<opt text= "0.83" correct="true">
 
Great!

</opt>

<opt text="0.87">

Are you sure you are calculating recall?


</opt>

<opt text="0.90">

Are you sure you are using Forward are your positive label?

</opt>

</choice>




**Question 3**    

What is the f1 score?

<choice id="3" >

<opt text="0.84"  correct="true">

Great!


</opt>

<opt text= "0.88">
 
Are you sure you are calculating f1? f1 = (2 * precision * recall) / (precision + recall)

</opt>

<opt text="0.82">

Are you sure you are calculating f1? f1 = (2 * precision * recall) / (precision + recall)


</opt>

<opt text="0.90">

Are you sure you are calculating f1? f1 = (2 * precision * recall) / (precision + recall)

</opt>


</choice>

</exercise>



<exercise id="7" title="True or False: Measurements">

**True or False?**     
*In spam classification, false positives are more damaging than false negatives (assume "positive" means the email is spam, "negative" means it's not).*

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False">

What would be worse, getting a spam email, or not getting an important email that was sent to junk?

</opt>

</choice>

**True or False**      
*In medical diagnosis, high recall is more important than high precision.*

<choice id="2">
<opt text="True"  correct="true">

Good job!

</opt>

<opt text="False"  >

We must identify as many of the positive values instead of assessing how many of our predicted positive labels are true.

</opt>

</choice >

</exercise>

<exercise id="8" title="Using Sklearn to Obtain Different Measurements">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's calculate some measurements from our basketball dataset from the previous question.

Tasks:   
- Import the precision, recall, f1 and classification report libraries. 
- Predict the values on `X_valid` using the `pipe_bb` and the `.predict()` function and save the result in an object named `predicted_y`.
- Using sklearn tools, calculate precision, recall and f1 scores and save them in the respective names `precision`, `recall`, and `f1`. Make sure you are comparing the true `y_valid` labels to the predicted labels. You will need to assign a positive label to the "Forward"(`F`) position. This can be specified in the `pos_label` of each function. Round each calculation to 3 decimal places.
- Print a classification report of all the measurements comparing `y_valid` and `predicted_y` and assigning the `target_names` argument to `["F", "G"]`. You can use the `digits` function to round all the calculations to 3 decimal places.


<codeblock id="07_08">

- Are you importing `precision_score`, `recall_score`, `f1_score` and  `classification_report`?
- Are you using the arguments `y_valid`, `predicted_y` and `pos_label="F"` for the scoring functions?
- Are you using the arguments `y_valid`, `predicted_y`  and `digits=3` for the `classification_report` function?


</codeblock>


**Question**    
Do the numbers in your classification report match the calculations you did using sklearn measurements?

<choice id="1" >
<opt text="Sure did!"  correct="true" >

Nice work!

</opt>

<opt text="No" >

Maybe give it another go above!

</opt>

</choice>

</exercise>

<exercise id="9" title="Multi-Class Measurements"  type="slides, video">
<slides source="module7/module7_09" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>


<exercise id="10" title="Using Sklearn to Obtain Different Measurements">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

This time with our basketball dataset we are predicting multiple positions, in the last one we used only "Forward" or "Guard" as our target labels and now we have 6 positions instead of 2!

Tasks:   
- Print a classification report of all the measurements comparing `y_valid` and `predicted_y`. You can use the `digits` function to round all the calculations to 3 decimal places.
- Using `zero_division=0` will suppress the warning. 


<codeblock id="07_10">

- Are you using the arguments `y_valid`, `predicted_y`,  and `digits=3` for the `classification_report` function?


</codeblock>


**Question 1**    
Using `G` as the positive label, what is the precision?

<choice id="1" >
<opt text="0.788"  correct="true" >

Nice work!

</opt>

<opt text="0.897" >

Are you looking at precision?

</opt>

<opt text="0.875">

Are you looking at the wrong label for the positive class?

</opt>

<opt text="0.778">

Are you looking at the wrong label for the positive class?

</opt>

</choice>


**Question 2**    

What is the recall of the model when we use `C` as the positive label? 

<choice id="2">

<opt text="0.00"   >

Not this time.

</opt>

<opt text="0.778" correct="true">

Nice!

</opt>

<opt text="0.875">

Are you looking at recall?

</opt>

<opt text="0.897">

Are you looking at the right positive label?

</opt>

</choice>


**Question 3**   
What is the weighted average precision measurement?

<choice id="3" >
<opt text="0.371"   >

This is the macro average precision

</opt>

<opt text="0.419" >

This is the macro average recall

</opt>

<opt text="0.555" correct="true">

Nice!

</opt>


<opt text="0.678">

This is the weighted average recall.

</opt>

</choice>

</exercise>


<exercise id="11" title= "Multi-class Questions">

<center><img src="/module7/multi-classQ.png"  width = "80%" alt="404 image" /></center>

For the next questions use the confusion matrix above and assume that **Forward** is the positive label. 

**Question 1**    

How many examples did the model correctly predict? 

<choice id="1">

<opt text="23" >

The correctly predicted examples are on the diagonal. 

</opt>

<opt text= "38">
 
The correctly predicted examples are on the diagonal. 

</opt>

<opt text="52" correct="true">

Got it! 

</opt>

<opt text="19">

The correctly predicted examples are on the diagonal. 

</opt>

</choice>


**Question 2**    

How many examples were incorrectly labeled as `G`? 

<choice id="2" >

<opt text="0" >

Look at the entire `G` column and disregard the correctly labeled (19) `G` examples.

</opt>

<opt text= "1">
 
Look at the entire `G` column and disregard the correctly labeled (19) `G` examples.

</opt>

<opt text="2" >

Look at the entire `G` column and disregard the correctly labeled (19) `G` examples.

</opt>

<opt text="3" correct="true">

Nice! 2 of these were incorrectly labeled as `G` when they should have been `F` and one should have been `G-F`. 

</opt>

</choice>


**Question 3**    

How many `F-C` labels were in the data?

<choice id="3">

<opt text="0" >

Are you looking at all the values in the `F-C` row and adding them up?

</opt>

<opt text= "1">
 
Are you looking at all the values in the `F-C` row and adding them up?

</opt>

<opt text="5" >

Are you looking at all the values in the `F-C` row and adding them up?

</opt>

<opt text="6"  correct="true">

Nice! 5 examples were incorrectly labeled as `F` and 1 labeled incorrectly as `C`. 

</opt>


</choice>

</exercise>



<exercise id="12" title="True or False: Measurements">

**True or False?**     
*The weighted average gives equal importance to all classes.*

<choice id="1" >
<opt text="True" >

It's the macro average that does this.

</opt>

<opt text="False" correct="true">

Killing it!

</opt>

</choice>

**True or False**      
*Using 1 target label as the positive class will make all other target labels negative.*

<choice id="2">
<opt text="True" correct="true" >

Good job!

</opt>

<opt text="False"  >

Classes are classified as binary for these measurements; either the target "positive" label or not. 
</opt>

</choice >

</exercise>

<exercise id="13" title="Imbalanced Datasets"  type="slides, video">
<slides source="module7/module7_13" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>

<exercise id="14" title="True or False: Unbalanced Data">

**True or False?**     
*When using `StratifiedKFold`, our data is no longer a random sample.*

<choice id="1" >
<opt text="True" correct="true">

Although this is true. Sometimes it's not a big problem.

</opt>

<opt text="False" >

Are you sure?

</opt>

</choice>

**True or False?**     
*If method A gets a higher accuracy than method B, that means its precision is also higher. .*

<choice id="2" >
<opt text="True" >

Precision is independent from accuracy. 

</opt>

<opt text="False" correct="true" >

Nice!

</opt>


</choice >

</exercise>


<exercise id="15" title="Balancing our Data in Action">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's bring back the Pokémon dataset that we've seen a few times. 

After splitting and inspecting the target column we see that this dataset is fairly unbalanced. 

In this case, our positive label is whether a Pokémon is **"legendary"**  or not. In our dataset, a value of 1 represents a legendary Pokémon and 0 is a non-legendary one.  

<codeblock id="pokemon_dist">

</codeblock>

Let's see how our measurements differ when we balance our datasets. 

Tasks:   
- Build a pipeline containing the column transformer and an SVC model with default hyperparameters. Fit this pipeline and name it `pipe_unbalanced`.
- Predict your values on the validation set and save them in an object named `unbalanced_predicted`.
- Using sklearn tools, print a classification report comparing the validation y labels to `unbalanced_predicted`. Set `digits=3`. 
- Next, build a pipeline containing the column transformer and an SVC model but this time setting `class_weight="balanced"` in the SVM classifier. Name this pipeline in an object called `pipe_balanced` and fit it on the training data. 
- Predict values on the validation set using `pipe_balanced` and save them in an object named `balanced_predicted`. 
- Print another classification report comparing the validation y labels to `balanced_predicted`. 



<codeblock id="07_15">
- Are you coding `unbalanced_predicted` as `make_pipeline(preprocessor, SVC())`. 
- Are you fitting on the training set?
- Are you building a classification report with `classification_report(y_valid, unbalanced_predicted, digits=2)`?
- Are you building `make_pipeline(preprocessor, SVC(class_weight="balanced"))` and fitting it?
- Are you predicting the values from the balanced pipeline using `pipe_balanced.predict(X_valid)` and naming it `balanced_predicted`?


</codeblock>


**Question 1**    
What happened with precision and recall when we balanced our class weights? 

<choice id="1" >
<opt text="Precision and recall both increased"   >

Are you looking at the wrong label for the positive class? `1` is our positive class in this situation.

</opt>

<opt text="Precision increased and recall decreased" >

Are you looking at the wrong label for the positive class? `1` is our positive class in this situation.

</opt>

<opt text="Precision and recall both decreased">

Are you looking at the wrong label for the positive class? `1` is our positive class in this situation.

</opt>

<opt text="Precision decreased and recall increased" correct="true">

Nice work! We sacrificed some precision for a better recall measurement. 

</opt>

</choice>


**Question 2**    

Did our accuracy increase or decrease when we used balanced class weights? 

<choice id="2">

<opt text="Increase"   >

Not this time.

</opt>

<opt text="Decrease" correct="true">

Nice!

</opt>

</choice>

</exercise>


<exercise id="16" title="Regression Measurements"  type="slides, video">
<slides source="module7/module7_16" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>


<exercise id="17" title= "Name That Measurement!">

**Question 1**    

Which measurement will have units which are the square values of the target column units?

<choice id="1">

<opt text="MSE" correct="true">

Great! 

</opt>

<opt text= "R<sup>2</sup>">
 
This won't have any units

</opt>

<opt text="RMSE">

This will have the same units are the target column

</opt>

<opt text="MAPE">

This is a percentage.

</opt>

</choice>


**Question 2**    

For which of the following is it possible to have negative values?

<choice id="2" >

<opt text="MSE" >

The lowest value here is 0! 

</opt>

<opt text= "R<sup>2</sup>" correct="true">
 
You got it. 

</opt>

<opt text="RMSE">

The lowest value here is 0! 

</opt>

<opt text="MAPE">

You cannot have a negative MAPE.

</opt>

</choice>




**Question 3**    

Which measurement is expressed as a percentage?

<choice id="3" >

<opt text="MSE" >

Not quite. 

</opt>

<opt text= "R<sup>2</sup>">
 
This will be a decimal and have a maximum value of 1.0. 

</opt>

<opt text="RMSE">

This is the square root of MSE and does not have relative values.

</opt>

<opt text="MAPE"  correct="true">

Go you!

</opt>

</choice>

</exercise>



<exercise id="18" title="True or False: Regression Measurements">

**True or False?**     
*We can still use recall and precision for regression problems but now we have other measurements we can use as well.*

<choice id="1" >
<opt text="True" >

We cannot use precision and recall anymore. 

</opt>

<opt text="False"  correct="true">

Great. 

</opt>

</choice>

**True or False**      
*A lower RMSE value indicates a better fit.*

<choice id="2">
<opt text="True"  correct="true">

Good job!

</opt>

<opt text="False"  >

Since this is an error measurement, we want a model that is resulting in less error.

</opt>

</choice >


**True or False**      
*In regression problems, calculating R<sup>2</sup>  using `r2_score()` and `.score()` (with default values) will produce the same results.*

<choice id="3">
<opt text="True"  correct="true">

Good job!

</opt>

<opt text="False"  >

`.score()` by default uses the R<sup>2</sup> measure.

</opt>

</choice >

</exercise>


<exercise id="19" title= 'Calculating "by hand"'>

**Question**    

Calculate the MSE from the values given below. 


|Observation | True Value | Predicted Value |
|------------|------------|-----------------|
|0           | 4          | 5               |
|1           | 12         | 10              |
|2           | 6          | 9               |
|3           | 9          | 8               |
|4           | 3          | 3               |



<choice id="1">

<opt text="15">

Don't forget to divide by the number of samples!

</opt>

<opt text= "8">
 
You need to sum up the *squared* differences. Not just the difference.

</opt>

<opt text="3" correct="true">

Nice work!

</opt>

<opt text="0">

This would be mean we perfectly predicted all the samples which we did not do in this case.

</opt>

</choice>

</exercise>

<exercise id="20" title="Calculating Regression Measurements">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's calculate some measurements from our basketball dataset this time predicting the players' salary. How well does our model do?

Tasks:   
- Import the mean squared error and R<sup>2</sup> libraries. 
- Calculate the MSE, RMSE, Rs<up>2</sup>, and MAP measurement by comparing the true values to what the model predicted on the validation set. Name the objects 
`mse_calc`, `rmse_calc`, `re_calc` and `mape_calc` respectively. 

<codeblock id="07_20">

- Are you importing `mean_squared_error` and `r2_score`?
- Are you using the arguments `y_valid` and `predict_valid` for your calculations?
- Are you using `np.sqrt()` on your `mse_calc` to calculate `rmse_calc`?
- Are you using `np.mean(np.abs((predict_valid - y_valid) / y_valid)) * 100.0` to calculate `mape_calc`?

pipe_bb.score(X_valid, y_valid))

</codeblock>

**Question**     
Would you deploy this model? 

<choice id="1" >
<opt text="Yes" >

Are you sure? These measurements are not looking good at all. 

</opt>

<opt text="No"  correct="true">

This model does worse than a DummyRegressor!

</opt>

</choice>

</exercise>

<exercise id="21" title="Passing Different Scoring Methods"  type="slides, video">
<slides source="module7/module7_21" shot="3" start="13:2011" end="14:1221">
</slides>

</exercise>

<exercise id="22" title= "True or False: Scoring with Cross-Validation">

**True or False?**     
*The `scoring` argument only accepts `str` inputs.*

<choice id="1" >
<opt text="True">

What about multiple scoring measures?

</opt>

<opt text="False"  correct="true">

Great!

</opt>

</choice>

**True or False**      
*We are limited to the scoring measures offered from sklearn.*

<choice id="2">
<opt text="True" >

We can use `make_scorer()` and use our own calculation. 

</opt>

<opt text="False" correct="true"  >

Good job!

</opt>

</choice >


**True or False**      
*If we specify the scoring method in `GridSearchCV` and `RandomizedSearchCV`, `best_param_`  will return the parameters with the highest specified measure.*

<choice id="3">
<opt text="True" correct="true" >

Cool!

</opt>

<opt text="False"  >

According to the source code, `best_param_` will return the parameter with the highest-scoring mean validation measure. 
This is why it's important to be mindful of if you are using an error measure or an accuracy one.

</opt>

</choice >

</exercise>




<exercise id="23" title="Scoring and Cross Validation">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's bring back the Pokémon dataset that we saw in exercise 15.  


<codeblock id="pokemon_dist">

</codeblock>

We've built our pipeline and looked at the classification reports but this time we want to do cross-validation and look at the scores from cross-validation of not just accuracy, but precision and recall as well.

Tasks:   
- Build a pipeline containing the column transformer and an SVC model and set `class_weight="balanced"` in the SVM classifier. Name this pipeline `main_pipe`.
- Perform cross-validation using `cross-validate` on the training split using the scoring measures accuracy, precision and recall.
- Save the results in a dataframe named `multi_scores`.




<codeblock id="07_23">
- Are you coding `main_pipe` as `make_pipeline(preprocessor, SVC())`. 
- Are you specifying `scoring = ['accuracy', 'precision', 'recall']` in your cross validation function? 
- Are you calling `cross_validate` on `main_pipe`, `X_train`, and `y_train`?
- Are you specifying `return_train_score=True` in `cross_validate`?
- Are you saving the result in a dataframe?



</codeblock>

</exercise>


<exercise id="24" title="What Did We Just Learn?" type="slides, video">
<slides source="module7/module7_end" shot="0" start="14:1222" end="14:3216">
</slides>

---
title: 'Module 3: Splitting, Cross-Validation and the Fundamental Tradeoff'
description:
  'This module will introduce you to why and how we split our data into a training and a test test, how cross validation works on our training data. We will explain the fundamental trade off as well as the golden rule of machine learning. '
prev: /module2
next: /module4
type: chapter
id: 3
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module2/module2_00" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="1" title="Splitting Data" type="slides,video">

<slides source="module3/module3_01" shot="3" start="0:003" end="1:54">
</slides>

</exercise>


<exercise id="2" title= "Splitting our data">

**Question 1**   
What are the 2 main groups we split our data into?

<choice id="1">

<opt text="Training and predicting data">

One of these may be right.

</opt>


<opt text= "Practice and testing data" >
 
One of these may be right.

</opt>

<opt text="Training and Testing data" correct="true">

Nice work!

</opt>

<opt text="Fiting and predicting data">

These are not the "official" names.

</opt>

</choice>



**Question 2**   
When do we split our data?

<choice id="2" >

<opt text="At the very begining, before we explore our data." correct="true">

Great!

</opt>

<opt text="After we explore our data, but before we make our model.">

It's important that we split our data before we do anything else.

</opt>


<opt text="After we train our model.">

This is a little too late. 

</opt>

<opt text="After we make any predictions.">

It's a better idea to do it as soon as possible.

</opt>

</choice>

**Question 3**   

Why do we split our data?

<choice id="3" >

<opt text="To increase our training accuracy." >

</opt>

<opt text="to help us generalize our model better." correct="true">

Great!

</opt>


<opt text="To decreasing training time.">

No quite but this may be a side effect. 

</opt>

</choice>

</exercise>


<exercise id="3" title="Decision Tree Outcome">

**True or False?**

_Splitting your data is randomized and you will get different results each time._

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False">

`train_test_split()` splits the given data in a randomized manner. 

</opt>

</choice>

**True or False: The following is an example of machine learning?**

_When using `train_test_split()`, you must specify both `test_size` and `train_size`._

<choice id="2">
<opt text="True" >

You only need to specify one of these arguments. 

</opt>

<opt text="False" correct="true">

Nice work! 

</opt>

</choice >

</exercise>


<exercise id="4" title="Splitting Data in Action">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's split our data using `train_test_split()` on our candy bars dataset.

Tasks:     

- Split the `X` and `y` dataframes into 4 objects: `X_train`, `X_test`, `y_train`, `y_test`. 
- Make the test set 0.2 (or the train set 0.8) and make sure to use `random_state=7`. 
- Build a model using `DecisionTreeClassifier()`. 
- Save this in an object named `model`. 
- Fit your model on the objects `X_train` and `y_train`.
- Evaluate the accuracy of the model using score on `X_train` and `y_train` save the values in an object named `train_score`.
- Repeat the above action but this time evaluate the accuracy of the model using score on `X_test` and `y_test` (which the model has never seen before) and save the values in an object named `test_score`. 

<codeblock id="03_04">

- Are you using `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)`? 
- Are using `DecisionTreeClassifier()`?
- Are you using the model named `model`?
- Are you calling `.fit(X_train, y_train)` on your model?
- Are you scoring your model using `model.score(X_train, y_train)` and `model.score(X_test, y_test)`?

</codeblock>



**Question 1**    
Which split performes better?

<choice id="1" >
<opt text="Training Data"   correct="true">

Nice job! 

</opt>

<opt text="Testing Data">

Great! The model predicted this one incorrectly.

</opt>

</choice>

</exercise>


<exercise id="5" title="Train, Validation and Test Split" type="slides,video">

<slides source="module3/module3_05" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="6" title= "Name that split!">

**Question 1**   
What data is trained on, predicted on and scored on? 

<choice id="1">

<opt text="Train"  correct="true">

Great! It's the only data that the model trains on.

</opt>


<opt text= "Validation" >
 
This doesn't see the training phase.

</opt>

<opt text="Test">

This 

</opt>

<opt text="Deployment">

This should be kept far far away from training and scoring! 

</opt>

</choice>



**Question 2**   
What is scored multiple times and predicted on multiple times but never is fitted?

<choice id="2" >

<opt text="Train" >

This data gets fitted, so it can't be trainning data.

</opt>


<opt text="Validate" correct="true">

Great!

</opt>

<opt text="Test">

It's a better idea to do it as soon as possible.

</opt>


<opt text="Deployment">

This is never scored. 

</opt>

</choice>

**Question 3**   

What is only ever scored once?

<choice id="3" >

<opt text="Train" >

This data gets scored many times.

</opt>


<opt text="Validate" >

This gets scored multiple times. 

</opt>

<opt text="Test" correct="true">

You got it!

</opt>


<opt text="Deployment">

This is never scored. 

</opt>

</choice>

</exercise>


<exercise id="7" title="Decision Tree Outcome">

**True or False?**

_Deployment data is used at the very end and only scored once._

<choice id="1" >
<opt text="True">  

Deployment data is used at the very end but it is never scored.

</opt>

<opt text="False" correct="true">

Nice job.

</opt>

</choice>

**True or False: The following is an example of machine learning?**

_Validation data is used to help tune hyperparameters._

<choice id="2">
<opt text="True" correct="true">

Nailed it!

</opt>

<opt text="False" >

This is a prime reason for having a validation set. 

</opt>

</choice >

</exercise>

<exercise id="8" title="Cross Validation" type="slides,video">

<slides source="module3/module3_08" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="9" title= "Name that split!">

**Question 1**   
What data is trained on, predicted on and scored on? 

<choice id="1">

<opt text="Train"  correct="true">

Great! It's the only data that the model trains on.

</opt>


<opt text= "Validation" >
 
This doesn't see the training phase.

</opt>

<opt text="Test">

This 

</opt>

<opt text="Deployment">

This should be kept far far away from training and scoring! 

</opt>

</choice>



**Question 2**   
What is scored multiple times and predicted on multiple times but never is fitted?

<choice id="2" >

<opt text="Train" >

This data gets fitted, so it can't be trainning data.

</opt>


<opt text="Validate" correct="true">

Great!

</opt>

<opt text="Test">

It's a better idea to do it as soon as possible.

</opt>


<opt text="Deployment">

This is never scored. 

</opt>

</choice>

**Question 3**   

What is only ever scored once?

<choice id="3" >

<opt text="Train" >

This data gets scored many times.

</opt>


<opt text="Validate" >

This gets scored multiple times. 

</opt>

<opt text="Test" correct="true">

You got it!

</opt>


<opt text="Deployment">

This is never scored. 

</opt>

</choice>

</exercise>


<exercise id="10" title="Decision Tree Outcome">

**True or False?**

_Deployment data is used at the very end and only scored once._

<choice id="1" >
<opt text="True">  

Deployment data is used at the very end but it is never scored.

</opt>

<opt text="False" correct="true">

Nice job.

</opt>

</choice>

**True or False: The following is an example of machine learning?**

_Validation data is used to help tune hyperparameters._

<choice id="2">
<opt text="True" correct="true">

Nailed it!

</opt>

<opt text="False" >

This is a prime reason for having a validation set. 

</opt>

</choice >

</exercise>

<exercise id="11" title="Cross Validation in Action">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's split our data using `train_test_split()` on our candy bars dataset.

Tasks:     

- Split the `X` and `y` dataframes into 4 objects: `X_train`, `X_test`, `y_train`, `y_test`. 
- Make the test set 0.2 (or the train set 0.8) and make sure to use `random_state=7`. 
- Build a model using `DecisionTreeClassifier()`. 
- Save this in an object named `model`. 
- Fit your model on the objects `X_train` and `y_train`.
- Evaluate the accuracy of the model using score on `X_train` and `y_train` save the values in an object named `train_score`.
- Repeat the above action but this time evaluate the accuracy of the model using score on `X_test` and `y_test` (which the model has never seen before) and save the values in an object named `test_score`. 

<codeblock id="03_11">

- Are you using `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)`? 
- Are using `DecisionTreeClassifier()`?
- Are you using the model named `model`?
- Are you calling `.fit(X_train, y_train)` on your model?
- Are you scoring your model using `model.score(X_train, y_train)` and `model.score(X_test, y_test)`?

</codeblock>



**Question 1**    
Which split performes better?

<choice id="1" >
<opt text="Training Data"   correct="true">

Nice job! 

</opt>

<opt text="Testing Data">

Great! The model predicted this one incorrectly.

</opt>

</choice>

</exercise>


<exercise id="21" title="What Did We Just Learn?" type="slides, video">
<slides source="module2/module3_end" shot="0" start="0:003" end="1:54">
</slides>
</exercise>

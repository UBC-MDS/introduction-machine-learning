---
title: 'Module 1: Introduction and Decision Trees'
description:
  'This chapter will explain the different branches of machine learning and introduce decision trees; a machine learning model used in supervised learning.'
prev: module0
next: /module2
type: chapter
id: 1
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module1/module1_00" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="1" title="What is Machine Learning" type="slides,video">

<slides source="module1/module1_01" shot="3" start="0:003" end="1:54">
</slides>

</exercise>


<exercise id="2" title="Examples of Machine">

**True or False: The following is an example of machine learning?**

_Diagnosing cancer from Magnetic resonance imaging_

<choice id="1" >
<opt text="True"  correct="true">

Great 

</opt>

<opt text="False">

Machine Learning can be use to attempt to diagnose cancer.

</opt>

</choice>

**True or False: The following is an example of machine learning?**

_Classifying images containing animals_

<choice id="2">
<opt text="True" correct="true">

Nice work! 

</opt>

<opt text="False">

If you look at example 2, in the slides you'll see the code we used to do this. 

</opt>

</choice >

**True or False: The following is an example of machine learning?**   
_Blowing out a candle_

<choice  id="3">
<opt text="True" >

No quite sure how this would work, but never say never! Technology is progressing everyday. 

</opt>

<opt text="I don't think we can use machine learning quite yet to " correct= "true">

Maybe one day we will be able to! Who knows. 

</opt>

</choice>

</exercise>


<exercise id="3" title="Types of Machine Learning" type="slides,video">

<slides source="module1/module1_03" shot="0" start="0:006" end="3:39">
</slides>

</exercise>



<exercise id="4" title=" Supervised vs. Unsupervised Learning">

**Given the following scenarios, would each example be considered supervised learning or unsupervised learning?**

**Example:**   
_Finding groups of similar properties in a real estate data set._

<choice id="1" >
<opt text="Supervised Learning">

Is there a "true number" of groups of similar properties? Are the groups known and defined?

</opt>

<opt text="Unsupervised Learning" correct="true">

Good job! This is an unsupervised learning example.

</opt>

</choice>

**Example:**    
_predicting real estate prices based on house features (number of rooms,past sales, etc.)_

<choice id="2">
<opt text="Supervised Learning" correct="true">

Nice work! Since we have examples with known value of real estate prices, we can use this predict real estate prices for homes we don't know the price on. 

</opt>

<opt text="Unsupervised Learning">

Do we have true corresponding values of what we are predicting with?

</opt>

</choice >

**Example:**   
_Detecting credit card fraud based on examples of fraudulent transactions._

<choice  id="3">
<opt text="Supervised Learning" correct= "true">

Great! Since we have examples with labels of "fraudulent" or "not fraudulent", we can detect if transactions with similar features to our examples are of the same nature. 

</opt>

<opt text="Unsupervised Learning" >

Do we have examples of the true corresponding value of what we are predicting?

</opt>

</choice>

**Example:**   
_Idenitfying groups of animals given features such as "number of legs", "wings/no wings", "fur/no fur", etc._

<choice  id="4">
<opt text="Supervised Learning" correct="true">

Nice work.

</opt>

<opt text="Unsupervised Learning" >

We know what the animal is suppose to be classfied as and we are hoping that with the given features, the model will be able to classify the animal.


</opt>
</choice>

**Example:**   
_Grouping articles on different topics from different news sources (something like Google News app)._


<choice  id="3">
<opt text="Supervised Learning">

Here we do not have a clear "correct" answer and therefor this is an example of unsupervised learning. 

</opt>

<opt text="Unsupervised Learning"  correct= "true">

Grouping is also called *clustering* which is an example of unsupervised learning. 

</opt>

</choice>

</exercise>



<exercise id="5" title="Classification vs. Regression" type="slides,video">

<slides source="module1/module1_05" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="6" title="Classification vs. Regression">


**Example:**  
_Predicting the price of a house based on features  such as  number of roomsand the year built._

<choice id="1">
<opt text="Classification">

Is the prediction a categorical or a numical value?

</opt>

<opt text="Regression" correct="true">

Good job! We are predicting a numerical value and therefore this is an example of regression.

</opt>

</choice>

**Example:**  
_Predicting if a house will sell or not based on features like the price of the house, number of rooms, etc._

<choice  id="2">
<opt text="Classification" correct="true">

Good job! We are predicting a categorical value (Sell/Not Sell) and therefore this is an example of classification.

</opt>

<opt text="Regression" >

Is the prediction a categorical or a numical value?

</opt>

</choice>

**Example:**  
_Predicting your grade in this course based on your grade in Programming in Python for Data Science._

<choice  id="3">
<opt text="Classification">

Is the prediction a categorical or a numical value?

</opt>

<opt text="Regression" correct="true">

Good job! We are predicting a numerical value (percent grade) and therefore this is an example of regression.

</opt>
</choice>

**Example:**  
_Predicting whether you should bicycle to work tomorrow based on the weather forecast._

<choice  id="4">
<opt text="Classification" correct="true">

Good job! We are predicting a categorical value (Bike/Not bike) and therefore this is an example of classification.

</opt>

<opt text="Regression">

Is the prediction a categorical or a numical value?

</opt>
</choice>
</exercise>

<exercise id="7" title="Tabular Data and Terminology" type="slides, video">

<slides source="module1/module1_07" shot="0" start="0:006" end="3:39">
</slides>

</exercise>



<exercise id="8" title="Terminology: Target">

Which is a synonym for ***targets***? 

<choice  id="1">

<opt text="Predictors" >

Not quite. You may want to have a read through of the definitions in this section. 

</opt>

<opt text="Records">

Not quite. You may want to have a read through of the definitions in this section. 

</opt>

<opt text="Outcomes" correct="true">

Good job!

</opt>

<opt text="Independent variables">

Not quite. You may want to have a read through of the definitions in this section.

</opt>

</choice>
</exercise>

<exercise id="9" title="Terminology: Features">

Which is **NOT** a synonym for ***features***? 

<choice  id="1">
<opt text="Inputs" >

Inputs is a synonym for features. You may want to have a read through of the definitions in this section. 

</opt>

<opt text="Records" correct="true">

Good job! Records is a synonym for examples, rows and samples

</opt>

<opt text="Predictors" >

Predictors is a synonym for features. You may want to have a read through of the definitions in this section. 

</opt>

<opt text="Independent variables">

Predictors is a synonym for features. You may want to have a read through of the definitions in this section. 

</opt>

</choice>

</exercise>


<exercise id="10" title="Describing a Dataset">

Let's make sure we understand all the components we use in a Dataset for machine learning. 

**Instructions:**

When you run a code exercise for the first time, it could take a bit of time for everything to load. 

**When you see `____` in a code exercise, replace it with what you assume to be the correct code. Run it and see if it you obtain the desired output. Submit your code to validate if you were correct.**

The packages you need will be loaded for you. 

- Print the `canbybar_df` object. 
- Save the dimensions of the dataframe in an object named `candybar_dim`.

<codeblock id="01_10">

- Are you using `.shape` to find the dimensions? 

</codeblock>

**Question:**  
How many features does the data have?

<choice  id="1">
<opt text="9">

This is the total number of columns, not the number of features

</opt>

<opt text="8" correct="true">

Yes! Good job!

</opt>

<opt text="25">

This is not the number of features.

</opt>

<opt text="10">

We don't include the index or the target as a feature.

</opt>
</choice>

**Question:**   
 How many examples does the data have?

<choice  id="2">

<opt text="9">

This is the total number of columns, not the number of examples.

</opt>

<opt text="8" >

This is the not the number of examples.

</opt>

<opt text="25" correct="true">

Well done!

</opt>

<opt text="26">
This is not the number of examples.

</opt>
</choice>

**Question:**   
Would this be considered classification or regression?

<choice  id="3">
<opt text="Classification" correct="true">

Great job!

</opt>

<opt text="Regression" >

What would we be predicting, a numerical value or categorical?

</opt>
</choice>
</exercise>

<exercise id="11" title="Baselines: Training a Model using Scikit-learn" type="slides">
<slides source="module1/module1_11">
</slides>
</exercise>

<exercise id="12" title="Fit or Predict">

**Do the following statements correspond to the `fit` or the `predict` function:**  

**Statement:**   
_Is called first (before the other one)._

<choice id="1">
<opt text="Fit" correct="true">

Great job! Training on training data must be done before predicting on new data. 

</opt>

<opt text="Predict" >

How can the model predict without education first?

</opt>
</choice>

**Statement:**  
_At least for decision trees, this is where most of the hard work is done._

<choice id="2">
<opt text="Fit" correct="true">

Great job! Training is more intensive then predicting for decision trees. 

</opt>

<opt text="Predict" >

Where do we teach the model? 

</opt>
</choice>

**Statement:**   
_Only takes `X` as an argument._

<choice id="3">
<opt text="Fit">

We need to make sure we give the model the correct labels so it can _learn_. 

</opt>

<opt text="Predict" correct="true">

Great job!

</opt>
</choice>

**Statement:**  
_In scikit-learn, we can ignore its output._

<choice id="4">
<opt text="Fit" correct="true">
 
Great job!

</opt>

<opt text="Predict">

You may want to look at the slides and see what each function delivers as an output. 

</opt>
</choice>
</exercise>

<exercise id="13" title="Separating our data">



**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's split up our data into  our features and target. 

Tasks:     
- Since we are attempting to predict the country availability of 
- Save the resulting dataframe as `pokemon_df`.
- It's a good idea to see what the [delimiter](https://github.com/UBC-MDS/MCL-DSCI-511-programming-in-python/blob/binder/data/pokemon-text.txt) is.
- Display the first 10 rows of `pokemon_df`.



<codeblock id="01_13">

- Make sure you are using the correct functions 
- Are you using the model named `model`?

</codeblock>
</exercise>

<exercise id="14" title="Decision Tree Splitting Rules" type="slides">
<slides source="module1/module1_14">
</slides>
</exercise>



<exercise id="14" title="What Did We Just Learn?" type="slides">
<slides source="module1_19">
</slides>
</exercise>
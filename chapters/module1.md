---
title: 'Module 1: Introduction and Decision Trees'
description:
  'This chapter will explain the different branches of machine learning and introduce decision trees; a machine learning model used in supervised learning.'
prev: module0
next: /module2
type: chapter
id: 1
---

<exercise id="1" title="Introduction + Supervised vs. Unsupervised learning" type="slides">

<slides source="module1_01">
</slides>

</exercise>

<exercise id="2" title="Scenario 1: Supervised vs. Unsupervised Learning">

Is finding groups of similar properties in a real estate data set an example of supervised learning or unsupervised learning?

<choice>
<opt text="Supervised Learning">

Is there a "true number" of groups of similar properties? Are the groups known and defined?

</opt>

<opt text="Unsupervised Learning" correct="true">

Good job! This is an unsupervised learning example.

</opt>

</choice>

</exercise>

<exercise id="3" title="Scenario 2: Supervised vs. Unsupervised Learning">

Is predicting real estate prices based on house features (number of rooms, learning from past sales, etc.) learning from past sales as examples supervised learning or unsupervised learning?

<choice>
<opt text="Supervised Learning" correct="true">

Nice work! Since we have examples with known value of real estate prices, we can use this predict real estate prices for homes we don't know the price on. 

</opt>

<opt text="Unsupervised Learning">

Do we have true corresponding values of what we are predicting with?

</opt>

</choice>

</exercise>

<exercise id="4" title="Scenario 3: Supervised vs. Unsupervised Learning">

Is detecting credit card fraud based on examples of fraudulent transactions an example supervised learning or unsupervised learning?

<choice>
<opt text="Supervised Learning" correct= "true">

Great! Since we have examples with labels of "fraudulent" or "not fraudulent", we can detect if transactions with similar features to our examples are of the same nature. 

</opt>

<opt text="Unsupervised Learning" >

Do we have examples of the true corresponding value of what we are predicting?

</opt>

</choice>

</exercise>

<exercise id="4" title="Scenario 4: Supervised vs. Unsupervised Learning">

Is identifying groups of animals given features such as "number of legs", "wings/no wings", "fur/no fur", etc. an example supervised learning or unsupervised learning?

<choice>
<opt text="Supervised Learning">

Not quite! Do we have predefined know groups that we are classifying?
</opt>

<opt text="Unsupervised Learning" correct="true">

Since we are clustering animals that are similar and there are no pre-defined groups, this is an example of unsupervised learning.

</opt>

</choice>

</exercise>

<exercise id="5" title="Classification vs. Regression" type="slides">

<slides source="module1_05">
</slides>

</exercise>

<exercise id="6" title="Scenario 1: Classification vs. Regression">

Is predicting the price of a house based on features like number of rooms an example of classification or regression?

<choice>
<opt text="Classification">

Is the prediction a categorical or a numical value?

</opt>

<opt text="Regression" correct="true">

Good job! We are predicting a numerical value and therefore this is an example of regression.

</opt>

</choice>

</exercise>

<exercise id="7" title="Scenario 2: Classification vs. Regression">

Is predicting if a house will sell or not based on features like the price of the house, number of rooms, etc. an example of classification or regression?

<choice>
<opt text="Classification" correct="true">

Good job! We are predicting a categorical value (Sell/Not Sell) and therefore this is an example of classification.

</opt>

<opt text="Regression" >

Is the prediction a categorical or a numical value?

</opt>

</choice>

</exercise>

<exercise id="8" title="Scenario 3: Classification vs. Regression">

Is predicting your grade in DSCI-571 based on past grades. an example of classification or regression?

<choice>
<opt text="Classification">

Is the prediction a categorical or a numical value?

</opt>

<opt text="Regression" correct="true">

Good job! We are predicting a numerical value (percent grade) and therefore this is an example of regression.

</opt>
</choice>
</exercise>

<exercise id="9" title="Scenario 4: Classification vs. Regression">

Is predicting whether you should bicycle to work tomorrow based on the weather forecast. an example of classification vs. regression?

<choice>
<opt text="Classification" correct="true">

Good job! We are predicting a categorical value (Bike/Not bike) and therefore this is an example of classification.

</opt>

<opt text="Regression">

Is the prediction a categorical or a numical value?

</opt>
</choice>
</exercise>

<exercise id="10" title="Tabular Data and Terminology" type="slides">

<slides source="module1_10">
</slides>

</exercise>

<exercise id="11" title="Terminology 1">

Which is a synonym for Target? 

<choice>
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

<exercise id="12" title="Terminology 2">

Which is not a synonym for features? 

<choice>
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

<exercise id="13" title="Describing a Dataset">

Let's make sure we understand all the components we use in a Dataset for machine learning. 

**Instructions:**

When you run a code exercise for the first time, it could take a bit of time for everything to load. 

**When you see `____` in a code exercise, replace it with what you assume to be the correct code. Run it and see if it you obtain the desired output. Submit your code to validate if you were correct.**

The packages you need will be loaded for you. 

- Print the `canbybar_df` object. 
- Save the feature names as a list named `candybar_feat`.
- Save the candybar names in a list named `candybar_names`.
- Save the dimensions of the dataframe as a tuple named `candybar_dim`.

<codeblock id="01_13">

- Are you sure you are saving `candybar_feat` and `candybar_names` as lists?
- Did you import the csv with the correct manner?

</codeblock>

Question: How many features does the data have?

<choice>
<opt text="9">

This is the total number of columns, not the number of features

</opt>

<opt text="8" correct="true">

Yes! Good job!

</opt>

<opt text="25">

This is not the number of features.

</opt>

<opt text="6">

This is not the number of features.

</opt>
</choice>
</exercise>

<exercise id="14" title="Describing a Dataset 2">

Question: How many examples does the data have?

<choice>

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
</exercise>

<exercise id="15" title="Describing a Dataset 3">
Question: Would this be considered classification or regression?

<choice>
<opt text="Classification" correct="true">

Great job!

</opt>

<opt text="Regression" >

What would we be predicting, a numerical value or categorical?

</opt>
</choice>
</exercise>

<exercise id="16" title="Training a Model using Scikit-learn" type="slides">
<slides source="module1_16">
</slides>
</exercise>

<exercise id="17" title="Fit or Predict 1">

Does this statement correspond to the `fit` or the `predict` function:   

_Is called first (before the other one)._

<choice>
<opt text="Fit" correct="true">

Great job! Training on training data must be done before predicting on new data. 

</opt>

<opt text="Predict" >

How can the model predict without education first?

</opt>
</choice>
</exercise>

<exercise id="18" title="Fit or Predict 2">

Does this statement correspond to the `fit` or the `predict` function: 

_At least for decision trees, this is where most of the hard work is done._

<choice>
<opt text="Fit" correct="true">

Great job! Training is more intensive then predicting for decision trees. 

</opt>

<opt text="Predict" >

Where do we teach the model? 

</opt>
</choice>
</exercise>

<exercise id="19" title="Fit or Predict 3">

Does this statement correspond to the `fit` or the `predict` function:   

_Only takes `X` as an argument._

<choice>
<opt text="Fit">

We need to make sure we give the model the correct labels so it can _learn_. 

</opt>

<opt text="Predict" correct="true">

Great job!

</opt>
</choice>
</exercise>

<exercise id="20" title="Fit or Predict 4">

Does this statement correspond to the `fit` or the `predict` function:   

_In scikit-learn, we can ignore its output._

<choice>
<opt text="Fit" correct="true">
 
Great job!

</opt>

<opt text="Predict">

You may want to look at the slides and see what each function delivers as an output. 

</opt>
</choice>
</exercise>

<exercise id="21" title="Building Your Model">

Let's try training and prediction our own model. 

**Instructions:**

When you run a code exercise for the first time, it could take a bit of time for everything to load. 

**When you see `____` in a code exercise, replace it with what you assume to be the correct code. Run it and see if it you obtain the desired output. Submit your code to validate if you were correct.**

<codeblock id="01_21">

- Make sure you are using the correct functions 
- Are you using the model named `model`?

</codeblock>
</exercise>

<exercise id="22" title="Decision Tree Splitting Rules" type="slides">
<slides source="module1_22">
</slides>
</exercise>

<exercise id="23" title="Calculating Gini Impurity">

Let's try calculating the Gini impurity  on our candybar dataset using the same `gini2` function from the module slides. The function has been imported for you. 

Let's make the split on the feature `peanuts` where anything greater than 0.5 is classified as `American`. Remember that you now have 2 groups and you need to calculate the impurity for both groups.  

* Don't forget to take into consideration the porportion of observations in each group!

**Instructions:**

When you run a code exercise for the first time, it could take a bit of time for everything to load. 

**When you see `____` in a code exercise, replace it with what you assume to be the correct code. Run it and see if it you obtain the desired output. Submit your code to validate if you were correct.**

The packages you need will be loaded for you. 
Save your Gini impurity as object `peanut_gini_impurity`

<codeblock id="01_23">

- Are you taking into consideration there are 2 gini calculations ( you will have to call `gini2 ` twice)?
- There are 6 observations of the 16 that have `peanuts` >= 0.5. Of those, 5 are of class `America` and 1 is of class `Canada`. 
- There are 10 observations of the 16 that have `peanuts` < 0.5. Of, those 3 are of class `America` and 7 is of class `Canada`.

</codeblock>
</exercise>

<exercise id="24" title="ML Model Parameters and Hyperparameters " type="slides">
<slides source="module1_24">
</slides>
</exercise>


<exercise id="25" title= "Summary Decision Tree Question 1">

What type of tree are Decision Trees?

<choice>
<opt text= "General Tree" >
 
How are the features split ?  

</opt>

<opt text="Binary Trees (2 children per node)." correct="true">

Nice work!

</opt>

<opt text="AVL Tree (self-balancing binary search tree).">

Does the tree need to be balanced? 

</opt>

</choice>
</exercise>

<exercise id="26" title="Summary Decision Tree Question 2">

Who choses the features that are split on at each node?

<choice>
<opt text= "Data Scientist" >
 
Where would we input this information?  

</opt>

<opt text="Model" correct="true">

Great!

</opt>

</choice>
</exercise>

<exercise id="27" title="Summary Decision Tree Question 3">

What is the depth of a decision stump? 

<choice>
<opt text= "1" correct="true">
 
You have been paying attention! Nice work! 

</opt>

<opt text="5" >

This is the default max depth of the decision tree classifier not the depth of a decision stump.

</opt>

<opt text="Whatever you set it as" >

Decision stumps are what make up a decision tree, stumps are not a hyperparameter

</opt>
</choice>
</exercise>

<exercise id="28" title="Summary Decision Tree Question 4">

Is the following Statement ***True*** or ***False***:

_The standard decision tree algorithm finds the optimal tree given a data set._

<choice>
<opt text= "True" >
 
For each node the model finds the optimal feature to split on but the nodes are chosen sequentially and it cannot choose the overall optimal tree. 

</opt>

<opt text="False" correct="true">

Great! Just because the model chooses the best feature to split on each node does not mean the total tree is the optimal one! 

</opt>

</choice>
</exercise>


<exercise id="29" title="Summary Decision Tree Question 5">

Is the following Statement ***True*** or ***False***:

_The same feature can be split on multiple times in a tree with depth > 1._

<choice>
<opt text= "True" correct="true">
 
Super! There can be multiple threshold splits used for the same feature. 

</opt>

<opt text="False" >

There can be multiple threshold splits used for the same feature. 

</opt>

</choice>
</exercise>

<exercise id="30" title="What Did We Just Learn?" type="slides">
<slides source="module1_30">
</slides>
</exercise>
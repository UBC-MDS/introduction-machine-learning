---
title: "Module 6: Preprocessing Categorical Variables and Sklearn's ColumnTransformer"
description:
  "This module will teach you different encoding methods for categorical variables (ordinal and one-hot encoding) and appropriately set them up. We will also introduce ColumnTransformer from the sklearn library and show you how to implement it for more complex pipelines."
prev: /module5
next: /module7
type: chapter
id: 6
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module6/module6_00" shot="0" start="11:4921" end="12:4509">
</slides>

</exercise>



<exercise id="1" title="" type="slides,video">

<slides source="module6/module6_01" shot="3" start="00:002" end="94:51">
</slides>

</exercise>

<exercise id="2" title= "">

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

<opt text="Fitting and predicting data">

These are not the "official" names.

</opt>

</choice>


**Question 2**    
When do we split our data?

<choice id="2" >

<opt text="At the very beginning, before we explore our data." correct="true">

Great!

</opt>

<opt text="After we explore our data, but before we make our model.">

We must split our data before we do anything else.

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

<opt text="to help us generalize our model better." >

Getting warmer but not quite. 

</opt>

<opt text="To decreasing training time.">

Not quite but this may be a side effect. 

</opt>


<opt text="To help us assess how well our model generalizes." correct="true">

Great!

</opt>


</choice>

</exercise>

<exercise id="3" title="">

**True or False?**      
_If you don't set random_state, splitting your data is randomized and you will get different results each time._

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False">

`train_test_split()` splits the given data in a randomized manner. 

</opt>

</choice>

**True or False**      
*When using `train_test_split()`, you must specify both `test_size` and `train_size`.*

<choice id="2">
<opt text="True" >

You only need to specify one of these arguments. 

</opt>

<opt text="False" correct="true">

Nice work! 

</opt>

</choice >

</exercise>

<exercise id="4" title="">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Tasks:     


<codeblock id="03_04">

- Are you ...?

</codeblock>


**Question**    
On which split does the decision tree perform better?

<choice id="1" >
<opt text="Training Data"   correct="true">

Nice job! 

</opt>

<opt text="Testing Data">

Maybe take a closer look?

</opt>

</choice>

</exercise>




<exercise id="30" title="What Did We Just Learn?" type="slides, video">
<slides source="module6/module6_end" shot="0" start="12:4510" end="13:2010">
</slides>
---
title: 'Module 5: Preprocessing numerical features, pipelines and hyperparameter optimization'
description:
  'This model will concentrate on the steps that need to be taken before building your model. Preperation through imputation and scaling is an important steps of model building and can be done using tools such as pipelines. Next we will explore how we can tune multiple hyperparameters at once using a process called Grid Search.'
prev: /module4
next: /module6
type: chapter
id: 5
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module5/module5_00" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="1" title="Why Preprocessing is Important" type="slides,video">

<slides source="module5/module5_01" shot="3" start="0:003" end="1:54">
</slides>

</exercise>

<exercise id="2" title= "Questions on Why">

**Question 1**  

Which model will still produce meaningful predictions with different scaled column values?

<choice id="1">

<opt text="Decision Trees" correct="true">

You are right! Decision Trees visit a single feature at a time unlike 𝑘-NN models, which calculate distances using all the features together. 

</opt>

<opt text= "𝑘-NN" >
 
Fantastic!

</opt>

<opt text="Dummy Classifier" >

Although this classifier is unaffected by different values in the feature columns, this is because this model isn't taking the feature values into it's prediction at all! This model is only predicting based on the primary target in the training set. 

</opt>

<opt text="SVM">

This works in a similar way to 𝑘-NN where distance is calculated and features are observe together and not independently of each other. 

</opt>

</choice>

**Question 2**   
*Complete the following statement*  
Preprocessing is done ____.  

<choice id="1" >

<opt text="To the model before training">

It is done before training but not to the model. 

</opt>

<opt text="To the data before training the model" correct="true">

Great!

</opt>

<opt text="To the model after training">

It's not done to the model or after training. 

</opt>

<opt text="To the data after training the model" >

You are half right but it's not done after training the model. 

</opt>

</choice>

</exercise>

<exercise id="3" title="Motivation True and False">

**True or False**     
_Columns will lower magnitudes compare to columns with higher magnitudes contribute are less important when making predictions._

<choice id="1" >
<opt text="True"  >

These types of models find examples that are most similar to the text example in the *training* set. 

</opt>

<opt text="False" correct="true">

Great! Just because a feature has smaller values does not mean it's less informative.

</opt>

</choice>

**True or False**     
*A model less sensitive to the scale makes it more robust.*

<choice id="2">
<opt text="True"  correct="true">

Nailed it!

</opt>

<opt text="False">

Models that are more sensitive to scale can be problematic. 

</opt>

</choice >

</exercise>


<exercise id="4" title="Preprocessing Questions">


**Question 1**  

`StandardScaler` is a type of what?

<choice id="1">

<opt text="Converter">

You are close but the terminology is not correct. 

</opt>

<opt text= "Categorizer" >
 
Not quite.

</opt>

<opt text="Model" >

We are not modeling with `StandardScaler` since it is *transforming* the data and not predicting on it. 

</opt>

<opt text="Transformer" correct="true">

Nice work!

</opt>

</choice>


**Question 2**  

What data does `StandardScaler` alter?

<choice id="2">

<opt text=" Training only">

It definitely alters the training set but that's not all. 

</opt>

<opt text= "Testing only" >
 
It does alter the test data but it doesn't stop there. 

</opt>

<opt text="Both training and testing" correct="true">

Nice work!

</opt>

<opt text="Nether training or testing ">

We need to scale our data though! 

</opt>

</choice>

</exercise>

<exercise id="5" title="Case Study: Preprocessing with Imputation" type="slides,video">
<slides source="module5/module5_05" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="6" title= "Imputation">


**Question 1**     

When do we need to imputate our data?


<choice id="1">

<opt text="When we have unreliable data.">

Not quite. 

</opt>

<opt text= "When we have missing data." correct="true">
 
Great!

</opt>

<opt text="Before we build all model." >

Not necessarily. It depends on the data we have. 

</opt>

<opt text="As a percaution to make sure our model is more robust.">

We will not be able to fit our model without imputation and so it's not not quite a precaution. 

</opt>

</choice>

**Question 2**   

If we have `NaN` values in our data, can we simply drop the column missing the data?

<choice id="2" >

<opt text="Yes, it won't make a difference."  >

What if it's a column that substantially helps the prediction?

</opt>

<opt text="Yes, if the mojority of the values are missing from the column" correct="true">

Great!

</opt>

<opt text="No droping the column will not solve the issue. ">

Dropping the column may solve the issue, however we could be losing a lot of important information. 

</opt>

<opt text="No, Never drop a column from the data.">

Dropping a column can sometimes be an appropriate method of removing `NaN` values. 

</opt>

</choice>

</exercise>

<exercise id="6" title="Imputation True or False">

**True or False**     
_`SimpleImputer` is a type of transformer._

<choice id="1" >
<opt text="True"  correct="true">

Yes! We are transforming the data!

</opt>

<opt text="False" >

Is our data transforming?

</opt>

</choice>


**True or False**     
_We can use `SimpleImputer` to impute values that are missing from numerical and categorical columns._

<choice id="2" >
<opt text="True"  correct="true">

Yes! We are can use `SimpleImputer` to impute both numerical and categorical columns.

</opt>

<opt text="False" >

This is true and we will touch on categorical columns in the next module!

</opt>

</choice>

</exercise>

<exercise id="7" title='Imputing in Action'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's take a look at a modified version of our basketball player dataset.  


First let's take a look at if and/or where we are missing any values 

Tasks:     

- Use `.describe()` or `.info()` to find if there are any values missing from the dataset. 
- Using some of the skills we learned in the previous <a href="https://prog-learn.mds.ubc.ca/en/module8" target="_blank">course</a> find the number of rows that contains missing values and save the total number of examples with missing values in an object named `num_nan`.       
*Hint: `.any(axis=1)` may come in handy here.* 

<codeblock id="05_07a">

- Are you using `X_train.info()`?
- Are you using `X_train.isnull().any(axis=1).sum()`?

</codeblock>



Now that we've identified the columns with missing values, let's use `SimpleImputer` to replace the missing value. 

Tasks:     
- Import the necessary library.
- Using `SimpleImputer`, replace the null values in the training and testing dataset with the medium value in each column.
- Save your transformed data in objects named `train_X_imp` and `test_X_imp` respectively. 
- Transform `X_train_imp` into a dataframe using the column and index labels from `X_train` and save it as `X_train_imp_df`.
- Check if `X_train_imp_df`  still has missing values.

<codeblock id="05_07b">



</codeblock>

</exercise>

<exercise id="8" title="Case Study: Preprocessing with Scaling" type="slides,video">
<slides source="module5/module5_08" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="9" title= "Scaling Multiple Choice">

**Question 1**  

What would happen if we didn't use `fill_diagonal()` when trying to find the closest example to an existing one?

<choice id="1">

<opt text="We would get the farthest example from the one we are trying to find instead of the closest.">

Not quite. 

</opt>

<opt text= "We would get itself as the closest example."  correct="true" >
 
Right, there is 0 distance from a point to itself. 

</opt>

<opt text="We would obtain the mean distance from all points to the current one." >

Unfortunately, the mean has nothing to do with why we fill the diagonals in.

</opt>

<opt text="We would get 0 examples." >

We would get an example but it would be the wrong one.

</opt>

</choice>

**Question 2**   

How many dimension does the input vector for `kneighbors()` need to be?

<choice id="2" >

<opt text="1" >

1d vectors will result in an error. 

</opt>

<opt text="2" correct="true">

Great!

</opt>

<opt text="3">

This will throw an error. 

</opt>

<opt text="It must be a pandas dataframe">

Close but you have the target value in the feature vector.

</opt>

</choice>

</exercise>

<exercise id="10" title="Scaling True or False">

**True or False**     
_Similar to decision trees, k-NNs finds a small set of good features._

<choice id="1" >
<opt text="True"  >

K-NNs use all the features!

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>

**True or False**     
_Finding the distances to a query point takes double the time as finding the nearest neighbour._

<choice id="2" >
<opt text="True"  >

This is completely made up!

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>

</exercise>

<exercise id="11" title='Practing Scaling'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's calculate the closet Pokémon in the training set to a Snoodle (our made-up Pokémon!

Snoodle	has the following feature vector. 

```out
[[53,  77,  43,  69,  80,  57,  379,  3]]
```
Which Pokémon in the training set, most resembles a Snoodle?

Tasks:     

- Create a model and name it `nn` (make sure you are finding the single closest Pokémon).
- Train your model on `X_train`.
- Predict your Pokémon using `kneighbors` and save it in an object named `snoodles_neighbour`.
- Which Pokémon (the name) is Snoodle most similar to? Save it in an object named `snoodle_name`.

<codeblock id="04_11">

- Are you importing ?
- Are you using ` NearestNeighbors(n_neighbors=1)`?
- Are you using `nn.fit(X_train)`?
- Are you using `nn.kneighbors(query_point)` ?
- Are you using `train_df.iloc[snoodles_neighbour[1].item()]['name']` to get the name of the closest Pokémon?

</codeblock>
</exercise>

<exercise id="12" title="Pipelines" type="slides,video">
<slides source="module5/module5_13" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="12" title="Hyperparameter Optimization Using Gridsearch" type="slides,video">
<slides source="module5/module5_13" shot="0" start="0:006" end="3:39">
</slides>

</exercise>



<exercise id="28" title="What Did We Just Learn?" type="slides, video">
<slides source="module5/module5_end" shot="0" start="0:003" end="1:54">
</slides>
</exercise>


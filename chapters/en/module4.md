---
title: 'Module 4: K-Nearest Neighbours'
description:
  'In this module we will cover a new model called 𝑘-nearest neighbours (also know as 𝑘-NNs) and how the Euclidean is calculated between 2 examples.'
prev: /module3
next: /module5
type: chapter
id: 4
---


<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module4/module4_00" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="1" title="Terminology" type="slides,video">

<slides source="module4/module4_01" shot="3" start="0:003" end="1:54">
</slides>

</exercise>

<exercise id="2" title= "Dimension Questions">

Use the following dataframe named `garden` to answer the next 2 questions. 

```out
               
       seeds   shape  sweetness   water-content      weight    fruit_veg
0      1        0        35          84               100        fruit
1      0        0        23          75               120        fruit
2      1        1        15          90              1360         veg
3      1        1         7          96               600         veg
4      0        0        37          80                 5        fruit
5      0        0        45          78                40        fruit  
6      1        0        27          83               450         veg
7      1        1        18          73                 5         veg
8      1        1        32          80                76         veg
9      0        0        40          83                65        fruit
```


**Question 1**  

We are trying to predict if each example is either a fruit or a vegetable. 
How many dimensions would dataset have?

<choice id="1">

<opt text="1">

Have you tried counting the columns in `garden` that are not the `target`? 

</opt>

<opt text= "5" correct="true">
 
Fantastic!

</opt>

<opt text="6" >

Are you including `fruit_veg` which is the target column? Or maybe you included the index?

</opt>

<opt text="7">

It's possible that you are including the index and target column `fruit_veg`. 

</opt>

</choice>


**Question 2**   


Which of the following would be the feature vector for example 0. 

<choice id="2" >

<opt text="<code>array([1,  0, 1, 1, 0, 0, 1, 1, 1, 0])<code>">

this is the values from the first column not values for the feature vector of example 0. 

</opt>

<opt text="<code>array([fruit,  fruit, veg, veg, fruit, fruit, veg, veg, veg, fruit])<code>">

this is only containing values of the target.

</opt>

<opt text="<code>array([1, 0, 35, 84, 100])<code>" correct="true">

Nice work!

</opt>

<opt text="<code>array([1, 0, 35, 84, 100,  fruit])<code>">

Close but you have the target value in the feature vector.

</opt>

</choice>

</exercise>

<exercise id="3" title="True and False Terminology">

**True or False**     
_Analogy-based models are finding examples from the test set that are most similar to the test example._

<choice id="1" >
<opt text="True"  >

These types of models find examples that are most similar to the text example in the *training* set. 


</opt>

<opt text="False" correct="true">

Great! They are finding examples in the training set, most similar to the test example. 

</opt>

</choice>

**True or False**     
*Feature vectors can only be of length 3 since we cannot visualize past that.*

<choice id="2">
<opt text="True" >

Feature vectors have no max value.

</opt>

<opt text="False" correct="true">

Nice work! 

</opt>

</choice >


**True or False**     
*A dataset with 50 dimensions is considered low dimensional.*

<choice id="2">
<opt text="True" correct="true">

Well done!

</opt>

<opt text="False" >

Dimensions up to 1000 are considered "low". 

</opt>

</choice >

</exercise>


<exercise id="4" title="Distances" type="slides,video">
<slides source="module4/module4_04" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="5" title= "Calculating Distances">

Given the following 2 feature vectors, which equation would calculate the Euclidean distance?

```
array([7, 0, 22, 11])
```


```
array([1, 0, 19, 9])
```


A) <img src="/module4/eq1.png"  width = "13%" alt="404 image" />


B) <img src="/module4/eq2.png"  width = "25%" alt="404 image" />


C) <img src="/module4/eq3.png"  width = "40%" alt="404 image" />


D) <img src="/module4/eq4.png"  width = "50%" alt="404 image" />




**Question 1**  

We are trying to predict if each example is either a fruit or a vegetable. 
How many dimensions would dataset have?

<choice id="1">

<opt text="A">

Not quite. 

</opt>

<opt text= "B" >
 
How many numbers are you performing subtraction on?

</opt>

<opt text="C" >

How many features are in each vector?

</opt>

<opt text="D"  correct="true">

Nice work.

</opt>

</choice>


**Question 2**   


What is the distance between the 2 vectors?

<choice id="2" >

<opt text="49"  correct="true">

You forgot to square root!

</opt>

<opt text="7">

Great!

</opt>

<opt text="6">

Nice work!

</opt>

<opt text="36">

Close but you have the target value in the feature vector.

</opt>

</choice>

</exercise>

<exercise id="6" title="Distance True or False">

**True or False**     
_Distance will always have a positive value._

<choice id="1" >
<opt text="True"  correct="true">

Yes! We are squaring all the differences which means distance can only be a positive value. 

</opt>

<opt text="False" >

Take a look at the equation we use to calculate Euclidean distance. 

</opt>

</choice>

</exercise>

<exercise id="7" title='Calculating Euclidean Distance by "Hand"'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's calculate the Euclidean distance between 2 examples in the Pokemon dataset without using Scikit-learn. 

Tasks:     

- Subtract the two first pokemon feature vectors and save it in an object name `sub_pk`.
- Square the difference and save it in an object named `sq_sub_pk`.
- Sum the squared difference from each dimension and save the result in an object named `sss_pk`.
- Finally, take the square root of the entire calculation and save it in an object named `pk_distance`.

<codeblock id="04_07">

- Are you importing `sqrt` from the `math` library?
- Are you using `X.iloc[1] - X.iloc[0]` to subtract the first 2 pokemon feature vectors?
- Are you using `**2` to square the difference??
- Are you using `.sum()` to sum the differences?
- Are you using `sqrt()` to square root the sum of squared differences?

</codeblock>

</exercise>

<exercise id="8" title='Calculating Euclidean Distance with Scikit-learn'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

This time, let's calculate the Euclidean distance between 2 examples in the Pokemon dataset using Scikit-learn. 

Tasks:     

- Import the necessary library.
- Calculate the Euclidean distance of the first 2 pokemon and save it in an object named pk_distance.

<codeblock id="04_08">

- Are you importing `euclidean_distances` from `sklearn.metrics.pairwise` 
- Are you making sure to use `euclidean_distances(X.iloc[:2])`
- Are you selecting the right value from the array using `[0,1]`
</codeblock>

</exercise>

<exercise id="9" title="Finding the Nearest Neighbour" type="slides,video">
<slides source="module4/module4_09" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="10" title= "Finding Neighbours Questions">


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

Nothing to do with mean here.

</opt>

<opt text="We would get 0 examples." >

We would get an example but it would be the wrong one.

</opt>

</choice>


**Question 2**   


What dimension does the input vector in `kneighbors()` must be?

<choice id="2" >

<opt text="1" >

1d vectors will result in an error. 

</opt>

<opt text="2" correct="true">

Great!

</opt>

<opt text="3">

This will throw and error. 

</opt>

<opt text="It must be a pandas dataframe">

Close but you have the target value in the feature vector.

</opt>

</choice>

</exercise>

<exercise id="11" title="Distance True or False">

**True or False**     
_Similar to decision trees, k-NNs finds a small set of good features._

<choice id="1" >
<opt text="True"  >

K-NNs use all the feature!

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

<exercise id="12" title='Calculating the Distance to a Query Point'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's calculate the closet pokemon in the training set to a Snoodle (our made up Pokemon!

Snoodle	has the following feature vector. 

```out
[[53,  77,  43,  69,  80,  57,  379,  3]]
```
Which Pokemon in the training set, most resembles a snoodle?


Tasks:     

- Create a model and name it `nn` (make sure you are finding the single closest pokemon).
- Train your model on `X_train`.
- Predict your pokemon using kneighbors and save it in an object named `snoodles_neighbour``.
- Which pokemon (the name) is Snoodle most similar to? Save it in an object named `snoodle_name`.

<codeblock id="04_12">

- Are you importing ?
- Are you using ` NearestNeighbors(n_neighbors=1)`?
- Are you using `nn.fit(X_train)`?
- Are you using `nn.kneighbors(query_point)` ?
- Are you using `train_df.iloc[snoodles_neighbour[1].item()]['name']` to get the name of the closest pokemon?


</codeblock>
</exercise>


<exercise id="13" title="𝑘 -Nearest Neighbours (𝑘-NNs) Classifier" type="slides,video">
<slides source="module4/module4_13" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="14" title= "Classifying Examples by Hand">

Consider this toy dataset:

<center><img src="/module4/Q14.png"  width = "40%" alt="404 image" /></center>

**Question 1**  

If 𝑘=1 , what would you predict for &nbsp; &nbsp;   <img src="/module4/ans14.png"  width = "8%" alt="404 image" /> &nbsp;&nbsp;&nbsp;?

<choice id="1">

<opt text="0">

the point (2, 2) is the closest to (0, 0).
 
</opt>

<opt text= "1"  correct="true" >
 
Right, the point (2, 2) is the closest to (0, 0) and it is categorized as 1. 


</choice>

**Question 2**  

If  𝑘=3 , what would you predict for &nbsp; &nbsp;   <img src="/module4/ans14.png"  width = "8%" alt="404 image" /> &nbsp;&nbsp;&nbsp;?

<choice id="2" >

<opt text="0" correct="true">

The points (2, 2), (5, 2) and (4, 3) are the closest to (0, 0).

</opt>

<opt text= "1" >
 
Right, there is 0 distance from a point to itself. 


</opt>

</choice>



</exercise>

<exercise id="15" title="K-NN Classifiers True or False">

**True or False**     
_The classification of the closest neighbour to the test example, always contributes the most to the prediction._

<choice id="1" >
<opt text="True">

Not always. You can select this as an option but it is not done like this by default.

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>


**True or False**     
_The `n_neighbors` hyperparameter must be less than the number of examples in the training set._

<choice id="2" >
<opt text="True" correct="true"  >

Nice work. 

</opt>

<opt text="False" >

You can't assign `n_neighbors` to a value greater than the possible number of examples in the trainning set. 

</opt>

</choice>

</exercise>

<exercise id="16" title='Predicting with a KNN-Classifier'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's try to classify some Pokemon from the Pokemon dataset. How well does our model do on the training data?

Tasks:     

- Create a `KNeighborsClassifier` model with `n_neighbors` equal to 5 and name it `model`.
- Train your model on `X_train` and`y_train` (Hint: you may want to use `.to_numpy()`).
- Score your model on the training set using `.score()` and save it in an object named `train_score`.
- Score your model on the test set using `.score()` and save it in an object named `test_score`.

<codeblock id="04_16">

- Are you importing `KNeighborsClassifier`?
- Are you using ` KNeighborsClassifier(n_neighbors=5)`?
- Are you using `model.fit(X_train, y_train.to_numpy())`?
- Are you using `model.score(X_train, y_train)` to find the training score?
- Are you using `model.score(X_test, y_test)` to find the test score?

</codeblock>
</exercise>


<exercise id="17" title="Choosing 𝑘 (n_neighbors)" type="slides,video">
<slides source="module4/module4_17" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="18" title= "Choosing K For Your Model">

Consider this graph:

<center><img src="/module4/Q18a.png"  width = "80%" alt="404 image" /></center>

**Question 1**  

What value of `n_neighbors` would you choose to train your model on? 

<choice id="1">

<opt text="0">

This is not a valid value for `n_neighbors`.
 
</opt>

<opt text= "1" >
 
Although this may have the highest training score, this does not have the highest cross-validation score. 

<opt text= "12"  correct="true" >
 
Nice work. 

</opt>

<opt text= "16" >
 
Almost. There is a value with a higher score

</opt>


<opt text= "29">
 
You shouldn't pick the highest `n_neighbors` without the cv-score being the highest.  

</opt>


</choice>

**Question 2**  

Up to which value of `n_neighbors` is there overfitting?

<choice id="2">

<opt text="The model never overfits">

When training score is greater than the CV score, the model is overfitting

</opt>

<opt text= "12" >
 
There is overfitting still occuring after this value. 

</opt>


<opt text= "26" >
 
Is training score greater that CV score after this value?

</opt>


<opt text= "29" correct="true">

Now it appears that the model is underfitting! 

</opt>


</choice>

</exercise>

<exercise id="19" title="Curse of Dimensionality True or False">

**True or False**     
_With  𝑘 -NN, setting the hyperparameter  𝑘  to larger values typically increase training score._

<choice id="1" >
<opt text="True">

Have you tred it out? It could be a good idea to see this in action!

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>


**True or False**     
_𝑘 -NN may perform poorly in high-dimensional space (say, d > 100)._

<choice id="2" >
<opt text="True" correct="true"  >

Nice work. 

</opt>

<opt text="False" >

Having more feature in some cases is less helpful to the model.

</opt>

</choice>

</exercise>

<exercise id="20" title='Hyperparameter tuning'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

In the last exercise we classified some Pokemon from the Pokemon dataset but we were not using the model that could have been the best! Let's try hyperparameter tuning.

First let's see which hyperparameter is the most optimal. 

Tasks:     

- Fill in the code for  a `for` loop that does the following:

  1. iterates over the values 1-30.
    - Builds a  `KNeighborsClassifier` model  with a `n_neighbors` equal to each iteration.
    - Uses `cross_validate` on the model with a `cv=10` and `return_train_score=True`.
    - Appends the depth value to the `n_neighbors` list in the dictionary `results_dict`.
    - Appends the `test_score` to the `mean_cv_score` list in the dictionary. 
    - Appends the `train_score` to the `mean_train_score` list in the dictionary. 
- We have given you code that wrangles this dictionary and transforms it into a state ready for plotting.
- Finish off by filling in the blank to create a line graph that plots the train and validation scores for each depth value. 
(Note: we have edited the limits of the y-axis so it's easier to read)

<codeblock id="04_20a">

- Are you importing `KNeighborsClassifier`?
- Are you using ` KNeighborsClassifier(n_neighbors=5)`?
- Are you using `model.fit(X_train, y_train.to_numpy())`?
- Are you using `cross_validate(model, X_train, y_train, cv=10, return_train_score=True)`?
- Are you using `alt.Chart(results_df).mark_line()` to create your plot?

</codeblock>



**Question 1**    
To which depth would you set your `n_neighbors` hyperparameter?

<choice id="1" >
<opt text="1" >

There are other depth values that have a higher cross-validation score that at this value. 

</opt>

<opt text="4" correct="true">

Nice work. This is where the score is at the highest for the validation set. 

</opt>

<opt text="8"   >

Are you sure this is the depth with the highest cross-validation score possible

</opt>

<opt text="17">

Are you sure this is the n_neighbors with the highest cross-validation score possible?

</opt>

</choice>

Tasks:     


Now that we have found a suitable value for `n_neighbors` let's build a new model and let this hyperparameter value. How well does your model do on the test data?

Tasks:     

- Build a model using `KNeighborsClassifier()` using the optimal `n_neighbors`. 
- Save this in an object named `model`. 
- Fit your model on the objects `X_train` and `y_train`.
- Evaluate the test score of the model using `.score()` on `X_test` and `y_test` and save the values in an object named `test_score` rounded to 4 decimal places.

<codeblock id="04_20b">

- Are using `KNeighborsClassifier(n_neighbors=?)`?
- Are you using the model named `model`?
- Are you calling `.fit(X_train, y_train)` on your model?
- Are you scoring your model using `model.score(X_test, y_test)`?
- Are you rounding to 4 decimal places?
- Are you calculating `test_score` as  `round(model.score(X_test, y_test), 4)` )

</codeblock>


**Question 1**    
Is the test score comparable with the cross-validation score that we obtained in the first part?

<choice id="1" >
<opt text="Yes"   correct="true">

Nice job! 

</opt>

<opt text="No">

Wouldn't you say within 3% is comparable here?

</opt>

</choice>

</exercise>


<exercise id="21" title="𝑘 -Nearest Neighbours Regressor" type="slides,video">
<slides source="module4/module4_21" shot="0" start="0:006" end="3:39">
</slides>

</exercise>


<exercise id="22" title= "Choosing K For Your Model">

Consider this toy dataset:

<center><img src="/module4/Q14.png"  width = "40%" alt="404 image" /></center>

**Question 1**  

If 𝑘=1 , what would you predict for &nbsp; &nbsp;   <img src="/module4/ans14.png"  width = "8%" alt="404 image" /> &nbsp;&nbsp;&nbsp;?

<choice id="1">

<opt text="0">

the point (2, 2) is the closest to (0, 0).
 
</opt>

<opt text= "1"  correct="true" >
 
Right, the point (2, 2) is the closest to (0, 0) and it is categorized as 1. 


</choice>

**Question 2**  

If  𝑘=3 , what would you predict for &nbsp; &nbsp;   <img src="/module4/ans14.png"  width = "8%" alt="404 image" /> &nbsp;&nbsp;&nbsp;?

<choice id="2" correct="true">

<opt text="0">

The points (2, 2), (5, 2) and (4, 3) are the closest to (0, 0).

</opt>

<opt text= "1" >
 
Right, there is 0 distance from a point to itself. 


</choice>



</exercise>


<exercise id="23" title='Hyperparameter tuning'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Tasks:     

- Create a `KNeighborsClassifier` model with `n_neighbors` equal to 5 and name it `model`.
- Train your model on `X_train` and`y_train` (Hint: you may want to use `.to_numpy()`).
- Score your model on the training set using `.score()` and save it in an object named `train_score`.
- Score your model on the test set using `.score()` and save it in an object named `test_score`.

<codeblock id="04_23">

- Are you importing `KNeighborsClassifier`?
- Are you using ` KNeighborsClassifier(n_neighbors=5)`?
- Are you using `model.fit(X_train, y_train.to_numpy())`?
- Are you using `model.score(X_train, y_train)` to find the training score?
- Are you using `model.score(X_test, y_test)` to find the test score?

</codeblock>
</exercise>


<exercise id="24" title="What Did We Just Learn?" type="slides, video">
<slides source="module4/module4_end" shot="0" start="0:003" end="1:54">
</slides>
</exercise>

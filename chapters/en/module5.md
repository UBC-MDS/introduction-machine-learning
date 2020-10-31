---
title: 'Module 5: Preprocessing Numerical Features, Pipelines and Hyperparameter Optimization'
description:
  'In this model, we will concentrate on the steps that need to be taken before building a model. Preparation through imputation and scaling is an important step of model building and can be done using tools such as pipelines. Next, we will explore automated hyperparameter optimization.'
prev: /module4
next: /module6
type: chapter
id: 5
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module5/module5_00" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="1" title="The Importance of Preprocessing" type="slides,video">

<slides source="module5/module5_01" shot="3" start="0:003" end="1:54">
</slides>

</exercise>

<exercise id="2" title= "Questions on Why">

**Question 1**      
Which model will still produce meaningful predictions with different scaled column values?

<choice id="1">

<opt text="Decision Trees" correct="true">

You are right! Decision Trees visit a single feature at a time, unlike 𝑘-NN models, which calculate distances using the features all together. 

</opt>

<opt text= "𝑘-NN" >
 
Fantastic!

</opt>

<opt text="Dummy Classifier" >

Although this classifier is unaffected by different values in the feature columns, this is because this model isn't taking the feature values into its prediction at all! This model is only predicting based on the primary target in the training set. 

</opt>

<opt text="SVM">

This works in a similar way to 𝑘-NN, where distance is calculated and features are observed together and not independently of each other. 

</opt>

</choice>

**Question 2**     
*Complete the following statement*  
Preprocessing is done ____.  

<choice id="2" >

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

You are half right, but it's not done after training the model. 

</opt>

</choice>

</exercise>

<exercise id="3" title="Motivation True and False">

**True or False**     
_Columns will lower magnitudes compared to columns with higher magnitudes are less important when making predictions._

<choice id="1" >
<opt text="True"  >

These types of models find examples that are most similar to the text example in the *training* set. 

</opt>

<opt text="False" correct="true">

Great! Just because a feature has smaller values does not mean it's less informative.

</opt>

</choice>

**True or False**     
*A model less sensitive to the scale of the data makes it more robust.*

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
When do we need to impute our data?

<choice id="1">

<opt text="When we have unreliable data.">

Not quite. 

</opt>

<opt text= "When we have missing data." correct="true">
 
Great!

</opt>

<opt text="Before we build all models." >

Not necessarily. It depends on the data we have. 

</opt>

<opt text="As a precaution to make sure our model is more robust.">

We will not be able to fit our model without imputation and so it's not quite a precaution. 

</opt>

</choice>

**Question 2**   
If we have `NaN` values in our data, can we simply drop the column missing the data?

<choice id="2" >

<opt text="Yes, it won't make a difference."  >

What if it's a column that substantially helps the prediction?

</opt>

<opt text="Yes, if the majority of the values are missing from the column" correct="true">

Great!

</opt>

<opt text=" No, dropping the column will not solve the issue. ">

Dropping the column may solve the issue, however, we could be losing a lot of important information. 

</opt>

<opt text="No, never drop a column from the data.">

Dropping a column can sometimes be an appropriate method of removing `NaN` values. 

</opt>

</choice>

</exercise>

<exercise id="7" title="Imputation True or False">

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

Yes! We can use `SimpleImputer` to impute both numerical and categorical columns.

</opt>

<opt text="False" >

This is true and we will touch on categorical columns in the next module!

</opt>

</choice>

</exercise>

<exercise id="8" title='Imputing in Action'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's take a look at a modified version of our basketball player dataset.  

First, let's take a look at if and/or where we are missing any values.

Tasks:     

- Use `.describe()` or `.info()` to find if there are any values missing from the dataset. 
- Using some of the skills we learned in the previous <a href="https://prog-learn.mds.ubc.ca/en/module8" target="_blank">course</a> find the number of rows that contains missing values and save the total number of examples with missing values in an object named `num_nan`.       
*Hint: `.any(axis=1)` may come in handy here.* 

<codeblock id="05_08a">

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

<codeblock id="05_08b">


</codeblock>

</exercise>

<exercise id="9" title="Case Study: Preprocessing with Scaling" type="slides,video">
<slides source="module5/module5_09" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="10" title= "Name that Scaling Method!">

**Question 1**    
Which scaling method will never produce negative values?

<choice id="1">

<opt text="Imputation">

This is a transformation to replace `NaN` values.

</opt>

<opt text= "Normalization"  correct="true" >
 
Perfect!

</opt>

<opt text="Standardization" >

This method will in fact produce negative values around a mean of 0. 

</opt>

<opt text="Both Normalization and Standardization" >

Although normalization is correct, standardization is not.

</opt>

</choice>

**Question 2**    
Which scaling method will never produce values greater than 1?

<choice id="2" >

<opt text="Imputation">

This is a transformation to replace `NaN` values.

</opt>

<opt text= "Normalization"  correct="true" >
 
Perfect!

</opt>

<opt text="Standardization" >

This method can produce values greater than 1 depending on the standard deviation of the values. 

</opt>

<opt text="Both Normalization and Standardization" >

Although normalization is correct, standardization can produce values greater than 1. 

</opt>

</choice>

**Question 3**    
Which scaling method will produce values where the range depends on the values in the data?

<choice id="2" >

<opt text="Imputation">

This is a transformation to replace `NaN` values.

</opt>

<opt text= "Normalization"  >
 
The range for values that have undergone Normalization will be 1.

</opt>

<opt text="Standardization" correct="true" >

This method's range depends on the standard deviation of the data. 

</opt>

<opt text="Both Normalization and Standardization" >

This time Standardization is correct but Normalization is not.

</opt>

</choice>

</exercise>

<exercise id="11" title="Scaling True or False">

**True or False**     
_Scaling is a form of transformation._

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False" >

Does it transform your data?

</opt>

</choice>

**True or False**     
_Scaling will always increase your training score._

<choice id="2" >
<opt text="True"  >

Scaling usually helps, but it is not guaranteed to increase the training score. 

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>

</exercise>

<exercise id="12" title='Practicing Scaling'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Now that we have a basketball dataset that no longer is missing any values, let's scale the features. 

First, let's scale using standardization. 

Tasks:     

- Import the necessary library for standardization.
- Build the transformer and name it `ss_scaler`.
- Fit and transform the data `X_train` and save the transformed feature vectors in objects named `X_train_scaled`.
- Transform `X_test` and save it in an object named `X_test_scaled`.
- Build a KNN classifier and name it `knn`.
- Fit your model on the newly scaled training data.
- Save the training score to 3 decimal places in an object named `ss_score`.

<codeblock id="05_12a">

- Are you importing `StandardScaler`?
- Are you using `ss_scaler.fit_transform(X_train)`?
- Are you using `model.fit(X_train, y_train.to_numpy())`?
- Are you using `ss_scaler.transform(X_test)`?
- Are you using `KNeighborsClassifier()` to create your model?
- Are you using `knn.fit(X_train_scaled, y_train)` to train your data?
- To obtain the training score are you using `knn.score(X_train_scaled, y_train).round(3)`?

</codeblock>


Let's try this again but this time, let's use normalization. 

Tasks:     

- Import the necessary library for normalization.
- Build the transformer and name it `mm_scaler`.
- Fit and transform the data `X_train` and save the transformed feature vectors in objects named `X_train_scaled`.
- Transform `X_test` and save it in an object named `X_test_scaled`.
- Build a KNN classifier and name it `knn`.
- Fit your model on the newly scaled training data.
- Save the training score to 3 decimal places in an object named `mm_score`.

<codeblock id="05_12b">

- Are you using `mm_scaler.fit_transform(X_train)`?
- Are you using `model.fit(X_train, y_train.to_numpy())`?
- Are you using `mm_scaler.transform(X_test)`?
- Are you using `pd.DataFrame(X_train_scaled, columns=X_train.columns)` to create your dataframe?
- Are you using `KNeighborsClassifier()` to create your model?
- Are you using `knn.fit(X_train_scaled, y_train)` to train your data?
- To obtain the training score are you using `knn.score(X_train_scaled, y_train).round(3)`?

</codeblock>

**Question**    
Which scaling transformation results in a better training score?

<choice id="1" >
<opt text="Standardization" >

Did standardization obtain a score greater than normalization? 

</opt>

<opt text="Normalization">

Was the score for normalization truly greater than that with normalization?

</opt>

<opt text="They obtained the same score" correct="true">

Great! They both scored  0.902 to 3 decimal places. 

</opt>

</choice>

</exercise>

<exercise id="13" title="Case Study: Pipelines" type="slides,video">
<slides source="module5/module5_13" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="14" title= "Pipeline Questions">

**Question 1**   
Which of the following steps cannot be used in a pipeline?

<choice id="1">

<opt text="Scaling">

We show an example of this in the slides.

</opt>

<opt text= "Model building"  >
 
We specify the model in the pipeline as the final step. 

</opt>

<opt text="Imputation" >

This is a transformation that can be added as a starting step in a pipeline.

</opt>

<opt text="Data Splitting"  correct="true">

We need to split our data into the training and testing splits *before* putting it into a pipeline. 

</opt>

</choice>

**Question 2**    
Why can't we fit and transform the training and test data together?

<choice id="2" >

<opt text="Because it would take a lot of time. ">

Think back to Module 3...

</opt>

<opt text= "It's violating the golden rule."  correct="true" >
 
Perfect!

</opt>

<opt text="It would result in an error." >

It wouldn't result in an error and would still execute (if your code was correct).

</opt>

<opt text="It would cause your model to underfit." >

Think of why we cannot mix the data together...

</opt>

</choice>

</exercise>

<exercise id="15" title="Pipeline True or False">

**True or False**     
_We have to be careful of the order we put each transformation and model in a pipeline._

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False" >

We need to make sure that the steps we would use outside a pipeline are reflected within it. 

</opt>

</choice>

**True or False**     
_Pipelines will fit and transform on both the training and validation folds during cross-validation._

<choice id="2" >
<opt text="True"  >

Using a `Pipeline` takes care of applying the `fit_transform` on the train portion and only `transform` on the validation portion in each fold.   

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>

</exercise>

<exercise id="16" title='Applying Pipelines'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Using our trusty basketball let's impute, scale and fit a model using a pipeline to see the results. 

Tasks:     

- Import the necessary library for building a pipeline.
- Build a pipeline named `bb_pipe` it should impute using `SimpleImputer` and a "median" strategy, scale using `StandardScaler` and build a `KNeighborsClassifier`.
- Cross-validate on `bb_pipe` using `X_train` and `y_train` and save the results in an object named `cross_scores`.
- Transform `cross_scores` to a dataframe, take the mean of each column and save the result in an object named mean_scores.

<codeblock id="05_16">

- Are you importing `Pipeline`?
- Are you using `SimpleImputer(strategy="median")` as the first step in the pipeline?
- Are you using `StandardScaler()` as a second step in the pipeline?
- Are you using `KNeighborsClassifier()` as the third step in the pipeline?
- Are you using `cross_validate(bb_pipe, X_train, y_train, return_train_score=True)` to cross-validate?
- Are you using `pd.DataFrame(cross_scores).mean()` to see your results?

</codeblock>

</exercise>

<exercise id="17" title="Automated Hyperparameter Optimization" type="slides,video">
<slides source="module5/module5_17" shot="0" start="0:006" end="3:39">
</slides>

</exercise>

<exercise id="18" title= "Exhaustive or Randomized Grid Search">

**Question 1**  
Which method will attempt to find the optimal hyperparameter for the data by searching every combination possible of hyperparameter values given?

<choice id="1">

<opt text="Exhaustive Grid Search"  correct="true">

We show an example of this in the slides.

</opt>

<opt text= "Randomize Grid Search"  >
 
We specify the model in the pipeline as the final step. 

</opt>

<opt text="Both" >

One of these search types is correct but not both. 

</opt>

</choice>

**Question 2**    
Which method is generally faster?

<choice id="2" >

<opt text="Exhaustive grid search">

Randomized grid search picks the best result more often than not and in a fraction of the time that it takes exhaustive grid search.

</opt>

<opt text= "Randomize grid search"   correct="true" >
 
Randomized grid search can be much faster than exhaustive grid search.

</opt>

<opt text="Both are relatively fast" >

Randomize grid search can be  much faster than exhaustive grid search.

</opt>

</exercise>

<exercise id="19" title="Hyperparameter Quick Questions">

**Question 1**     
_If I want to search for the most optimal hyperparameter values among 3 different hyperparameters each with 3 different values how many trials of cross-validation would be needed?_

<choice id="1" >
<opt text="6"  >

Each hyperparameter would be checked with each value. The calculation would be 3 * 3 * 3. 

</opt>

<opt text="9" >

Each hyperparameter would be checked with each value. The calculation would be  3 * 3 * 3. 

</opt>

<opt text="27" correct="true">

Each hyperparameter would be checked with each value. The calculation would be  3 * 3 * 3. 
In this case, it would be 3 * 3 * 3 =27 

</opt>

<opt text="81" >

Each hyperparameter would be checked with each value. The calculation would be  3 * 3 * 3. 


</opt>

</choice>

**True or False**     
_Automated hyperparameter optimization can only be used for multiple hyperparameters._

<choice id="2" >
<opt text="True"  >

Grid search can be used for a single parameter too, however now it's just searching in 1 dimension. 

</opt>

<opt text="False" correct="true" >

Great work!

</opt>

</choice>

</exercise>

<exercise id="20" title='Using Automated Hyperparameter Optimization in Action!'>

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Now that we have built a pipeline in the last interactive exercises, let's pair that with grid search to optimize our hyperparameters. 

Tasks:     

- Import the necessary grid search libraries. 
- Using the pipeline provided, create a grid of parameters to search over named `param_grid`. Search over the values 1, 5, 10, 20, 30, 40, and 50 for the hyperparameter `n_neighbors` and 'uniform' and 'distance' for the hyperparameter `weights` (make sure to call them appropriately). 
- Use `GridSearchCV` to hyper-parameter tune using cross-validate equal to 10 folds. Make sure to specify the arguments `verbose=1` and `n_jobs=-1`. Name the object grid_search. 
- Transform `cross_scores` to a dataframe, take the mean of each column and save the result in an object named mean_scores.
- Find the best hyperparameter values and save them in an object named `best_hyperparams`. Make sure to print these results.
- Create a new model using the best parameters found in grid search and save this model as `best_model_pipe`. Don't forget to train it after.
- Lastly, score your model on the test set and save your results in an object named `bb_test_score`.

<codeblock id="05_20">

- Are you importing `Pipeline`?
- Are you specifying `knn__n_neighbors` and `knn__weights` in `param_grid` and specifying the hyperparameter values in a list?
- Are you using `GridSearchCV(bb_pipe, param_grid, cv=10, verbose=1, n_jobs=-1)` and remembering to fit it?
- Are you using `grid_search.best_params_` to find the most optimal hyperparameter values?
- Are you using `grid_search.best_estimator_` to build the model with these optimal hyperparameters?
- Are you using training `best_model_pipe
- Are you using `best_model_pipe.score(X_test, y_test)` to calculate your test score?

</codeblock>

**Question 1**  
How many times is a model trained in the problem above?

<choice id="1">

<opt text= "9"  >
 
Think multiplications instead of addition.

</opt>

<opt text="14" >

This is how many times cross-validation is called. 

</opt>

<opt text="140">

You are extremely close but what about fitting the new model?

</opt>

<opt text="141"  correct="true">

Great! We cannot forget the model we made for our test set!

</opt>

</choice>

**Question 2**  
What was the optimal value for the hyperparameter `weights`? 

<choice id="2">

<opt text="uniform"  correct="true">

Great! 

</opt>

<opt text= "distance"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

</choice>

**Question 3**  
What is the optimal value for the hyperparameter `n_neighbors`? 

<choice id="3">

<opt text= "1"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

<opt text= "5"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

<opt text= "10"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

<opt text= "20"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

<opt text= "30"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

<opt text="40"  correct="true">

You got it!

</opt>

<opt text= "50"  >
 
Are you sure? Take a look at what `grid_search.best_params_` returns.

</opt>

</choice>

</exercise>

<exercise id="21" title="What Did We Just Learn?" type="slides, video">
<slides source="module5/module5_end" shot="0" start="0:003" end="1:54">
</slides>


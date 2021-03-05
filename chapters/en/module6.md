---
title: "Module 6: Preprocessing Categorical Variables"
description:
  "This module will teach you different encoding methods for categorical variables (ordinal and one-hot encoding) and appropriately set them up. We will also introduce ColumnTransformer and CountVectorizer from the sklearn library and show you how to implement them."
prev: /module5
next: /module7
type: chapter
id: 6
---

<exercise id="0" title="Module Learning Outcomes"  type="slides, video">

<slides source="module6/module6_00" shot="0" start="11:4921" end="12:4509">
</slides>

</exercise>



<exercise id="1" title="Categorical Variables: Ordinal Encoding" type="slides,video">

<slides source="module6/module6_01" shot="3" start="00:005" end="07:3712">
</slides>

</exercise>

<exercise id="2" title= "Categorical Variables">

```
           name    colour    location    seed   shape  sweetness   water-content  weight  popularity
0         apple       red     canada    True   round     True          84         100      popular
1        banana    yellow     mexico   False    long     True          75         120      popular
2    cantaloupe    orange      spain    True   round     True          90        1360      neutral
3  dragon-fruit   magenta      china    True   round    False          96         600      not popular
4    elderberry    purple    austria   False   round     True          80           5      not popular
5           fig    purple     turkey   False    oval    False          78          40      neutral
6         guava     green     mexico    True    oval     True          83         450      neutral
7   huckleberry      blue     canada    True   round     True          73           5      not popular
8          kiwi     brown      china    True   round     True          80          76      popular
9         lemon    yellow     mexico   False    oval    False          83          65      popular

```

**Question 1**    

What would be the unique values given to the categories in the `popularity` column, in we transformed it with ordinal encoding?

<choice id="1">

<opt text="<code>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]</code>">

There are multiples of some of these values

</opt>

<opt text= "<code>[0, 1, 2]</code>"  correct="true">
 
Nice!

</opt>

<opt text="<code>[1, 2, 3]</code>">

Do we start labeling at 1?

</opt>

<opt text="<code>[0, 1, 2, 3]</code>">

Do we have 4 unique values?

</opt>

</choice>


**Question 2**    

Does it make sense to be doing ordinal transformations on the `colour` column?

<choice id="2" >

<opt text="Yes" >

Is yellow more red than green?

</opt>

<opt text="No" correct="true">

Good work!

</opt>

</choice>

</exercise>

<exercise id="3" title="True or False: Ordinal Encoding">

**True or False?**     
Whenever we have categorical values, we should use ordinal encoding.

<choice id="1" >
<opt text="True" >

Do all categorical features have an order? For example, if we had fruit, is a kiwi closer to a banana than a strawberry?

</opt>

<opt text="False" correct="true">

Great!

</opt>

</choice>

**True or False**      
If we include categorical values in our feature table, `sklearn` will throw an error.

<choice id="2">
<opt text="True" correct="true" >

Nice work! 

</opt>

<opt text="False">

Do categorical variables make sense to `sklearn`?

</opt>

</choice >

</exercise>

<exercise id="4" title="Try Ordinal Encoding Yourself!">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

We've seen our basketball dataset but have only used the features `salary`, `weight` and `height`. This time, let's look at the `country` column and transform it. 

Tasks:   
- Import the necessary library.
- Build an ordinal encoder that uses a `dtype` of `int` and name it `ordinal_encoder`. 
- Fit on `X_column`, transform it and save the results in an object named `country_encoded`. 


<codeblock id="06_04">

- Are you importing `OrdinalEncoder`?
- Are you building `OrdinalEncoder` and using `dtype=int`?
- Are you fitting the transformer?


</codeblock>


**Question**    
Which country corresponds with group 5?

<choice id="1" >
<opt text="USA"   >

Maybe take a closer look?

</opt>

<opt text="Croatia" correct="true">

Nice!

</opt>

<opt text="Greece">

Maybe take a closer look?

</opt>


<opt text="Egypt">

Maybe take a closer look?

</opt>

</choice>

</exercise>

<exercise id="5" title="Categorical Variables: One-Hot Encoding" type="slides,video">

<slides source="module6/module6_05" shot="3" start="07:4406" end="13:2606">
</slides>

</exercise>


<exercise id="6" title= "One-Hot Encoding Questions">

**Question**     
If we one hot encoded the `shape` column, what datatype would be the output of after using `transform`?

<choice id="1">

<opt text="NumPy array" correct="true">

You got it!

</opt>

<opt text= " Pandas Dataframe" >
 
Do we get labels with the output?

</opt>

<opt text="Pandas Series" >

Are we getting multiple columns in the output?

</opt>

<opt text="Dictionary">

Not quite. 

</opt>

</choice>

</exercise>

<exercise id="7" title= "One-Hot Encoding - Output">

Refer to the dataframe to answer the following question.
```
           name   colour location   seed  shape  sweetness  water_content  weight
0         apple      red   canada   True  round       True             84     100
1        banana   yellow   mexico  False   long       True             75     120
2    cantaloupe   orange    spain   True  round       True             90    1360
3  dragon-fruit  magenta    china   True  round      False             96     600
4    elderberry   purple  austria  False  round       True             80       5
5           fig   purple   turkey  False   oval      False             78      40
6         guava    green   mexico   True   oval       True             83     450
7   huckleberry     blue   canada   True  round       True             73       5
8          kiwi    brown    china   True  round       True             80      76
9         lemon   yellow   mexico  False   oval      False             83      65
```

<br>

**Question**   
Which of the following outputs in the result of one-hot encoding the `shape` column?

A) 

```
array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 0]])
```

B)

```
array([[0, 0, 1],
       [1, 0, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 1, 0]])
```

C)

```
array([[0, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0]])
```

D) 

```
array([[0],
       [5],
       [0],
       [3],
       [0],
       [0],
       [3],
       [0],
       [5],
       [3],
       [1],
       [4],
       [3],
       [2]])

```


<choice id="1" >

<opt text="A"  >

Are you sure it's the correct dimensions?

</opt>

<opt text="B" correct="true">

Great!

</opt>

<opt text="C">

How many unique values are there in the column `seed` 

</opt>

<opt text="D">

This is a single column. Are you sure that's what you want?

</opt>

</choice>

</exercise>

<exercise id="8" title="One Hot encoding True or False">

**True or False**     
_One-hot encoding a column with 5 unique categories will produce 5 new transformed columns._

<choice id="1" >
<opt text="True"  correct="true">

Yes! We are transforming the data into new columns!

</opt>

<opt text="False" >

How is our data transforming?

</opt>

</choice>

**True or False**     
_The values in the new transformed columns after one-hot encoding, are all possible integer or float values._

<choice id="2" >
<opt text="True"  >

How many options are there for each columns? Does it become a binary value now?

</opt>

<opt text="False" correct="true">

Great! The values become binary and only two possible values are in the columns now; 0, or 1. 

</opt>

</choice>

</exercise>

<exercise id="9" title="Encoding - One-Hot Style!">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Last time we ordinal encoded the `country` column from our basketball dataset but now we know that this isn't the best option. This time, instead let's one-hot encode this feature. 

Tasks:    

- Import the necessary library.
- Build a one-hot encoder that uses a `dtype` of `int` and `sparse=False`. Name it `one_hot_encoder`. 
- Fit on `X_column`, transform it and save the results in an object named `country_encoded`. 

<codeblock id="06_09">

- Are you importing `OneHotEncoder`?
- Are you building `OneHotEncoder` and using `dtype=int` and setting `sparse=False`?
- Are you fitting the transformer?

</codeblock>


**Question**    
How many columns will `country_encoded` have?

<choice id="1" >
<opt text="1"   >

This is the case for ordinal encoding. Try using `country_encoded.shape`. 

</opt>

<opt text="5">

This is just what you see. Try using `country_encoded.shape`. 

</opt>

<opt text="21" >

Try using `country_encoded.shape`. 

</opt>

<opt text="23"  correct="true">

Nice!

</opt>

</choice>


</exercise>


<exercise id="10" title="ColumnTransformer" type="slides,video">

<slides source="module6/module6_10" shot="3" start="13:3325" end="28:0905">
</slides>

</exercise>

<exercise id="11" title= "Transforming Columns with ColumnTransformer">


Refer to the dataframe to answer the following question.
```
       colour   location    shape   water_content  weight
0       red      canada      NaN         84          100
1     yellow     mexico     long         75          120
2     orange     spain       NaN         90          NaN
3    magenta     china      round        NaN         600
4     purple    austria      NaN         80          115
5     purple    turkey      oval         78          340
6     green     mexico      oval         83          NaN
7      blue     canada      round        73          535
8     brown     china        NaN         NaN        1743  
9     yellow    mexico      oval         83          265
```

<br>

**Question 1**   
How many categorical columns are there and how many numeric?

 

<choice id="1" >

<opt text="5 categoric columns and 0 numeric columns"  >

There are some numeric colums?

</opt>

<opt text="2 categoric columns and 3 numeric columns" >

Are you missing a categorical columns?

</opt>

<opt text="3 categoric columns and 2 numeric columns" correct="true">

Great!

</opt>

<opt text="1 categoric columns and 4 numeric columns">

Are you counting numerical columns that I am not?

</opt>

</choice>


**Question 2**   
What transformations are being done to both numeric and categorical columns?

<choice id="2" >

<opt text="Scaling"  >

This is used on numeric columns.

</opt>

<opt text="Imputation" correct="true">

Great!

</opt>

<opt text="One-hot encoding">

This is only used on categorical columns.

</opt>

<opt text="Pipeline">

Pipeline isn't a transformer.

</opt>

</choice>

</exercise>

<exercise id="12" title="Transforming True or False">

**True or False**     
_If there are missing values in both numeric and categorical columns, we can specify this in a single step in the main pipeline._

<choice id="1" >
<opt text="True"  >

We specify the transformation in each column type pipeline before we use them as inputs for `ColumnTransformer`. 

</opt>

<opt text="False" correct="true">

Nailed it!

</opt>

</choice>

**True or False**     
_If we do not specify `remainder="passthrough"` as an argument in `ColumnTransformer`, the columns not being transformed will be dropped ._

<choice id="2" >
<opt text="True"  correct="true">

You got it! Without this, any columns left alone will be removed from your features. 

</opt>

<opt text="False" >

Are you certain? Slide 7 has some information regarding this. 
</opt>

</choice>

</exercise>

<exercise id="13" title=" Your Turn with Column Transforming">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Let's now start doing transformations and working with them with our basketball dataset. 

We've provided you with the numerical and categorical features, it's your turn to make a pipeline for each and then use `ColumnTransformer` to transform them. 

We have a regression problem this time where we are attempting to predict a player's salary.


Tasks:     
- Import the necessary library.
- Create a pipeline for the numeric features. It should have the first step as simple imputation using `strategy="median"` and the second step should be using `StandardScaler`.  Name this pipeline `numeric_transformer`.
- Create a pipeline for the categorical features. It should also have 2 steps. The first is imputation using `strategy="most_frequent"`. The second step should be one-hot encoding with `handle_unknown="ignore"`. Name this pipeline `categorical_transformer`. 
- Make your column transformer named `col_transformer` and specify the transformations on `numeric_features` and `categorical_features` using the appropriate pipelines you build above. 
-Create a main pipeline named `main_pipe` which preprocesses with `col_transformer` followed by building a `KNeighborsRegressor` model.
- The last step is performing cross-validation using our pipeline.



<codeblock id="06_13">

- Are you importing the right library?
- Are you using `SimpleImputer(strategy="median")` for numerical imputation? 
- Are you naming your steps?
- Are you using `SimpleImputer(strategy="most_frequent")` for categorical imputation?
- Are you using one-hot encoding?
- Are you naming the steps in `ColumnTransformer` and specifying `numeric_transformer` with `numeric_features` and `categorical_transformer` with `categorical_features`?
- Is the first step in your main pipeline calling `col_transformer`?
- Are you calling `main_pipe` in `cross_validate()`?

</codeblock>

</exercise>


<exercise id="14" title="Make - Pipelines & Column Transformers" type="slides,video">

<slides source="module6/module6_14" shot="3" start="28:1901" end="41:55">
</slides>

</exercise>


<exercise id="15" title= "Making pipelines">

Use the diagram below to answer the following questions.

```
Pipeline(
    steps=[('columntransformer',
               ColumnTransformer(
                  transformers=[('pipeline-1',
                                  Pipeline(
                                    steps=[('simpleimputer',
                                             SimpleImputer(strategy='median')),
                                           ('standardscaler',
                                             StandardScaler())]),
                      ['water_content', 'weight', 'carbs']),
                                ('pipeline-2',
                                  Pipeline(
                                    steps=[('simpleimputer',
                                             SimpleImputer(fill_value='missing',
                                                                strategy='constant')),
                                           ('onehotencoder',
                                             OneHotEncoder(handle_unknown='ignore'))]),
                      ['colour', 'location', 'seed', 'shape', 'sweetness',
                                                   'tropical'])])),
         ('decisiontreeclassifier', DecisionTreeClassifier())])
                
```  

**Question 1**   
How many columns are being transformed in `pipeline-1`?
 

<choice id="1" >

<opt text="0"  >

Are you counting the right thing? Look above `pipeline-2`.

</opt>

<opt text="2" >


Are you counting the right thing? Look above `pipeline-2`.

</opt>

<opt text="3" correct="true">

Great! They are ` ['water_content', 'weight', 'carbs']

</opt>

<opt text="6">

Are you counting the columns for `pipeline-2` by accident?

</opt>

</choice>


**Question 2**   
Which pipeline is transforming the categorical columns?

<choice id="2" >

<opt text="pipeline-1"  >

This is using `StandardScaler` so it is likely transforming numeric columns. Also, pipeline-2 is using `OneHotEncoder`.

</opt>

<opt text="pipeline-2" correct="true">

Great!

</opt>

</choice>



**Question 3**   
What model is the pipeline fitting on?

<choice id="3" >

<opt text="<code>SVC</code>"  >

This is used on numeric columns.

</opt>

<opt text="<code>KNeighborsClassifier</code>" >

</opt>

<opt text="<code>DummyClassifier</code>">

This is only used on categorical columns.

</opt>

<opt text="<code>DecisionTreeClassifier</code>" correct="true">

Great!

</opt>

</choice>

</exercise>

<exercise id="16" title="Transforming True or False">

**True or False**     
*`Pipeline()` is the same as `make_pipeline()` but  `make_pipeline()` requires you to name the steps.*

<choice id="1" >
<opt text="True"  >

`Pipeline()` requires you to name the steps whereas `make_pipeline()` does not. 

</opt>

<opt text="False" correct="true">

Nailed it! It's the other way round! `Pipeline()` requires you to name the steps. 

</opt>

</choice>

</exercise>


<exercise id="17" title="Making Pipelines with make_pipeline()">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

Let's try to redo exercise 13, but this time let's use `make_pipeline()` and `make_column_transformer`. 


Tasks:   
- Import the necessary library.
- For all pipelines, make sure to use `make_pipeline()` where possible.
- Create a pipeline for the numeric features. It should have the first step as simple imputation using `strategy="median"` and the second step should be using `StandardScaler`.  Name this pipeline `numeric_transformer`. 
- Create a pipeline for the categorical features. It should also have 2 steps. The first is imputation using `strategy="most_frequent"`. The second step should be one-hot encoding with `handle_unknown="ignore"`. Name this pipeline `categotical_transformer`. 
- Make your column transformer named `col_transformer` by using `make_column_transformer()`and specify the transformations on `numeric_features` and `categorical_features` using the appropriate pipelines you build above. 
- Create a main pipeline named `main_pipe` which preprocesses with `col_transformer` followed by building a `KNeighborsRegressor` model.
- The last step is performing cross-validation using our pipeline.



<codeblock id="06_17">

- Are you importing the right library?
- Are you using `SimpleImputer(strategy="median")` for numerical imputation? 
- Are you naming your steps?
- Are you using `SimpleImputer(strategy="most_frequent")` for categorical imputation?
- Are you using one-hot encoding?
- Are you  specifying `numeric_transformer` with `numeric_features` and `categorical_transformer` with `categorical_features` in `make_column_transformer`?
- Is the first step in your main pipeline calling `col_transformer`?
- Are you calling `main_pipe` in `cross_validate()`?

</codeblock>

</exercise>



<exercise id="18" title="Handeling Categorical Features: Binary, Ordinal and More" type="slides,video">

<slides source="module6/module6_18" shot="3" start="42:05" end="54:5011">
</slides>

</exercise>



<exercise id="19" title= "Transforming Categorical Features">

Use the diagram below to answer the following questions.

```
   colour  tropical location  carbs   seed  shape        size  water_content  weight
0      red     False   canada      6   True  round      small             84     100
1   yellow      True   mexico     12  False   long        med             75     120
2   orange     False    china      8   True  round      large             90    1360
3  magenta     False    china     18   True  round      small             96     600
4   purple     False   mexico     11  False  round      small             80       5
5   purple     False   canada      8  False   oval        med             78      40
6    green      True   mexico     14   True   oval        med             83     450
7     blue     False   canada      6   True  round      large             73       5
8    brown      True    china      8   True  round      large             80      76
9   yellow      True   mexico      4  False   oval        med             83      65
```


**Question 1**   
 On which column would you  use `OneHotEncoder(sparse=False, dtype=int, drop="if_binary")`?

<choice id="1" >

<opt text="<code>colour</code>"  >

Is this column binary?

</opt>

<opt text="<code>location</code>" >

Is this column binary?

</opt>

<opt text="<code>seed</code>" correct="true">

Great! This column is binary!

</opt>

<opt text="<code>size</code>">


Is this column binary?

</opt>

</choice>

</choice>



**Question 2**   
What model would you use ordinal encoding with?

<choice id="2" >

<opt text="<code>colour</code>"  >

Does colour have ordinality? 

</opt>

<opt text="<code>location</code>" >

Is location ordinal?

</opt>

<opt text="<code>seed</code>" >

This column contains binary values.

</opt>

<opt text="<code>size</code>" correct="true">

Great!

</opt>

</choice>

</exercise>

<exercise id="20" title="Categorical True or False">

**True or False**     
_It's important to be mindful of the consequences of including certain features in your predictive model._

<choice id="1" >
<opt text="True"  correct="true">

Great!

</opt>

<opt text="False">

It's important to remember the systems you build are going to be used in some applications.

</opt>

</choice>

</exercise>


<exercise id="21" title="Transforming the Fertility Dataset">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

For this question, we will be using a dataset from assignment 1. 

Here is the requested citation:
_David Gil, Jose Luis Girela, Joaquin De Juan, M. Jose Gomez-Torres, and Magnus Johnsson. Predicting seminal quality with artificial intelligence methods. Expert Systems with Applications, 39(16):12564 “ 12573, 2012_

We will be making pipelines and transforming our features appropriately. 


First, let's take a look at our dataset and the features. 

*Disclaimer: Normally we should be investing more time to fully understand the data we are analyzing. We should be checking the unique values, using `.describe()`  and `.info()` to really get an idea of our features before deciding which transformations we want to apply.* 

<codeblock id="fertility">

</codeblock>

Tasks:   
- What are the numeric features? Add them to a list named `numeric_features`.
- What are the binary features? Add them to a list named `binary_features`.
- What are the ordinal features? Add them to a list named `ordinal_features`.
- What are the rest of the categorical features? Add them to a list named `categorical_features`.
- Order the values in `high_fevers_last_year` and name the list `fever_order`. The options are 'more than 3 months ago', 'less than 3 months ago' and 'no'.
- Order the values in `smoking_habit` and name the list `smoking_order`. The options are 'occasional', 'daily' and 'never'.
- Order the values in `freq_alcohol_con` and name the list `alcohol_order`. The options are 'once a week', 'hardly ever or never', 'several times a week', 'several times a day' and 'every day'.
- There are several pipelines already made for you. Designate `numeric_transformer` to the numerical transformer, `categorical_transformer` to the transformer that is not transforming binary or ordinal features, `binary_transformer` to the transformer of binary features, and `ordinal_transformer1`, `ordinal_transformer2` and `ordinal_transformer3` to the transformer of columns `high_fevers_last_year`, `smoking_habit` and `freq_alcohol_con` respectively. 
- Fill in the associated gaps in the column transformer named `preprocessor`. 
- Build a main pipeline using `KNeighborsClassifier` and name the object `main_pipe`.
- Cross-validate and see the results.

<codeblock id="06_21">

- Are you ordering the ordinal values correctly? 
- Do you have 3 binary features?
- Are you naming the pipelines correctly?

</codeblock>

</exercise>




<exercise id="22" title="Text Data" type="slides,video">

<slides source="module6/module6_22" shot="3" start="54:5913" end="66:5120">
</slides>

</exercise>


<exercise id="23" title= "Text Data Questions">

**Question 1**   
What is the size of the vocabulary for the examples below?

```
X = [ "Take me to the river",
    "Drop me in the water",
    "Push me in the river",
    " dip me in the water"]

```



<choice id="1" >

<opt text="20"  >

This is the total number of words.

</opt>

<opt text="5" >

This is for the first observation only.

</opt>

<opt text="10" correct="true">

Got it!

</opt>

<opt text="11">

Are you counting a word twice?

</opt>

</choice>


**Question 2**   
Which of the following is not a hyperparameter of `CountVectorizer()`?

<choice id="2" >

<opt text="<code>binary</code>"  >

This is a hyperparameter of `CountVectorizer()`.

</opt>

<opt text="<code>max_features</code>" >

This is a hyperparameter of `CountVectorizer()`.

</opt>

<opt text="<code>vocabulary</code>" correct="true">

This is not a hyperparameter but an attribute!

</opt>

<opt text="<code>ngram_range</code>">

This is a hyperparameter of `CountVectorizer()`.

</opt>

</choice>

</exercise>

<exercise id="24" title="Text Data True or False">

**True or False**     
*As you increase the value for the `max_features` hyperparameter of `CountVectorizer`, the training score is likely to go up.*

<choice id="1" >
<opt text="True"  correct="true">

Great! The model is becoming more complex.

</opt>

<opt text="False">

Increasing the value of `max_features` means we include each and every word from the training data in the dictionary and the training score is likely to go up.

</opt>

</choice>

**True or False**     
_If we encounter a word in the validation or the test split that's not available in the training data, we'll get an error._

<choice id="2" >
<opt text="True">

If the word isn't in the dictionary, we would just ignore the word.

</opt>

<opt text="False" correct="true">

We have one for each category type and 1 main pipeline.

</opt>

</choice>

</exercise>

<exercise id="25" title="CountVectorizer with Disaster Tweets">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_

We are going to bring in a new dataset for you to practice on. 

This dataset contains a text column containing tweets associated with disaster keywords and a target column denoting whether a tweet is about a real disaster (1) or not (0). (<a href="https://www.kaggle.com/vstepanenko/disaster-tweets" target="_blank"><b>Source</b></a>)

In this question, we are going to explore how changing the value of `max_features` affects our training and cross-validation scores.

Tasks:

- Import `CountVectorizer`.
- Split the dataset into the feature table `X` and the target value `y`. `X` will be the single column `text` from the dataset wheras `target` will be your `y`. 
- Split your data into your training and testing data using a text size of 20% and a random state of 7. 
- Make a pipeline with `CountVectorizer` as the first step and `SVC()` as the second. Name the pipeline `pipe`. 
- Perform RandomizedSearchCV using the parameters specified in `param_grid` and name the search `tweet_search`.
- Don't forget to fit your grid search.
- What is the best `max_features` value? Save it in an object name `tweet_feats`.
- What is the best score? Save it in an object named `tweet_val_score`.
- Score the optimal model on the test set and save it in an object named `tweet_test_score`.

NOTE: This may take a few minutes to produce an output. Please be patient.

<codeblock id="06_25">

- Are you splitting using `train_test_split()`
- Are you using `make_pipeline(CountVectorizer(), SVC())`?
- Are you using `RandomizedSearchCV()` and calling `pipe` and `param_grid` as the first 2 arguments?
- Are you naming the randomized grid search `tweet_search`?
- Are you fitting `tweet_search`?
- Are you using `tweet_search.best_params_['countvectorizer__max_features']` to get the optimal number of features?
- Are you using `tweet_search.best_score_` to get the best validation score?
- Are you using `tweet_search.score(X_test, y_test)` to get the test score?

</codeblock>

</exercise>



<exercise id="26" title="What Did We Just Learn?" type="slides, video">
<slides source="module6/module6_end" shot="0" start="12:4510" end="13:2010">
</slides>
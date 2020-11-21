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

<slides source="module6/module6_01" shot="3" start="00:002" end="94:51">
</slides>

</exercise>

<exercise id="2" title= "Categorical Variables">

```
           name    colour    location    seed   shape  sweetness   water-content  weight
0         apple       red     canada    True   round     True          84         100
1        banana    yellow     mexico   False    long     True          75         120
2    cantaloupe    orange      spain    True   round     True          90        1360
3  dragon-fruit   magenta      china    True   round    False          96         600
4    elderberry    purple    austria   False   round     True          80           5
5           fig    purple     turkey   False    oval    False          78          40
6         guava     green     mexico    True    oval     True          83         450
7   huckleberry      blue     canada    True   round     True          73           5
8          kiwi     brown      china    True   round     True          80          76
9         lemon    yellow     mexico   False    oval    False          83          65

```

**Question 1**    

What would be the unique values givent to th values in the column `location`, in we transformed it with ordinal encoding?

<choice id="1">

<opt text="<code>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]</code>">

There are multiples of some of these values

</opt>

<opt text= "<code>[0, 1, 2, 3, 4, 5]</code>"  correct="true">
 
Nice!

</opt>

<opt text="<code>[1, 2, 3, 4, 5, 6]</code>">

Do we start labelling at 1?

</opt>

<opt text="<code>[0, 1, 2, 3, 4, 5, 6]</code>">

Do we have 7 unique values?

</opt>

</choice>


**Question 2**    

Does it make sense to be doing ordinal transformations on this column?

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

Do all categorical has an order to them? For example if we had fruit, is a kiwi closer to a banana than a strawberry?

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

We've seen our basketball dataset but have only used the features `salary`, `weight` and `height`. This times let's look at the `country` column and transform it. 

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
Which country corresponds with group 21?

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

<slides source="module6/module6_05" shot="3" start="00:002" end="94:51">
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
Which of the following outputs in the result of one-hot encoding the shape column?

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
_The unique values in the new transformed columns after one-hot encoding because all possible integer or float values._

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
- Build a one-hot encoder that uses a `dtype` of `int` and `sparse=True`. Name it `one_hot_encoder`. 
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

<slides source="module6/module6_10" shot="3" start="00:002" end="94:51">
</slides>

</exercise>

<exercise id="11" title= "Transforming Columns with ColumnTransformer">


Refer to the dataframe to answer the following question.
```
       colour   location   seed    shape   sweetness  water_content  weight
0       red      canada    True              True          84         100
1     yellow     mexico    False   long      True          75         120
2     orange     spain     True              True          90           
3    magenta     china     True    round     False                    600
4     purple    austria    False             True          80         115
5     purple    turkey     False   oval      False         78         340
6     green     mexico     True    oval      True          83           
7      blue     canada     True    round     True          73         535
8     brown     china      True              True                    1743  
9     yellow    mexico     False   oval      False         83         265
```

<br>

**Question 1**   
How many categorical columns are there and how many numeric?

 

<choice id="1" >

<opt text="5 categoric columns and 3 numeric columns"  >

Are you sure it's the correct dimensions?

</opt>

<opt text="3 categoric columns and 5 numeric columns" >



</opt>

<opt text="5 categoric columns and 2 numeric columns" correct="true">

Great!

</opt>

<opt text="3 categoric columns and 4 numeric columns">

This is a single column. Are you sure that's what you want?

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
_If there are missing values in both numeric and categorical columns we can specify this in a single step in the main pipeline._

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

We have a regression problem this time where we are attempting to predict a players salary.


Tasks:     
- Create a pipeline for the numeric featurers. It should have the first step as simple imputation using 
- Create a pipeline for the categorical features. It should also have 2 steps. The first is imputation using `strategy="median"`. The second step should be one-hot encoding with `handle_unknown="ignore"`. 


<codeblock id="06_13">

- Are you ...?

</codeblock>

</exercise>


<exercise id="14" title="Make - Pipelines & Column Transformers" type="slides,video">

<slides source="module6/module6_14" shot="3" start="00:002" end="94:51">
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
How many columns are being transformed in the first pipeline?
 

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

This is using `StandardScaler` so it is likely transforming numeric columns. Also pipeline-2 is using `OneHotEncoder`.

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
_`Pipeline()` is the same as `make_pipeline()` but  `make_pipeline()` requires you to name the steps._

<choice id="1" >
<opt text="True"  >

`Pipeline()` requires you to name the steps whereas `make_pipeline()` does not. 

</opt>

<opt text="False" correct="true">

Nailed it! It's the other way round! `Pipeline()` requires you to name the steps. 

</opt>

</choice>

**True or False**     
_`make_pipeline()` can be called before `make_column_transformer`._

<choice id="2" >
<opt text="True"  correct="true">

Nice work!

</opt>

<opt text="False" >

We can first make seperate transformation pipelines for our different columns and then we can use `make_column_transformer`. 

</opt>

</choice>

</exercise>


<exercise id="17" title="">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Tasks:     


<codeblock id="">

- Are you ...?

</codeblock>

</exercise>



<exercise id="18" title="Handeling Categorical Features: Binary, Ordinal and more" type="slides,video">

<slides source="module6/module6_18" shot="3" start="00:002" end="94:51">
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
Which column would you use `OneHotEncoder(sparse=False, dtype=int, drop="if_binary")`?
 

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


**Question 2**   
Which column would you group into bigger categories?

<choice id="2" >

<opt text="<code>colour</code>"  correct="true">

This has several categories and many with only 1 value. 

</opt>

<opt text="<code>location</code>" >

This only has 3 unique values.

</opt>

<opt text="<code>seed</code>" >

This column already contains binary values.

</opt>

<opt text="<code>size</code>">

This only has 3 unique values.

</opt>



</choice>



**Question 3**   
What model would you use ordinal encoding with?

<choice id="3" >

<opt text="<code>colour</code>"  >

Does colour have ordinality. 

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

It's important to rememberthe systems you build are going to be used in some applications.

</opt>

</choice>

**True or False**     
_If we have numeric, ordinal, binary and regular categorical features, we will need to call `make_pipeline()` 5 times to build a model._

<choice id="2" >
<opt text="True"  correct="true">

Nice work! We have one for each category type and 1 main pipeline.

</opt>

<opt text="False" >

We have one for each category type and 1 main pipeline.

</opt>

</choice>

</exercise>


<exercise id="21" title="">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Tasks:     


<codeblock id="">

- Are you ...?

</codeblock>

</exercise>




<exercise id="22" title="Text Data" type="slides,video">

<slides source="module6/module6_22" shot="3" start="00:002" end="94:51">
</slides>

</exercise>


<exercise id="23" title= "Text Data Questions">

**Question 1**   
What is the size of the vocabulary for the examples below?

```
X = [ "Take me to the river",
    "Drop me in the water.",
    "Push me in the river,",
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
Which transformer created the sparse matrix below for the data in Question 1?

```
array([[0, 1, 1, 1, 0],
       [1, 1, 0, 1, 1],
       [1, 1, 1, 1, 0],
       [1, 1, 0, 1, 1]])
```


<choice id="2" >

<opt text="<code>CountVectorizer(binary=True)</code>"  >

How many columns should this matrix have given the vocabulary above?

</opt>

<opt text="<code>CountVectorizer()</code>" >

Think about some parameters that may need to be set.

</opt>

<opt text="<code>CountVectorizer(binary=True, max_features=5)</code>" correct="true">

You've been paying attention!

</opt>

<opt text="<code>CountVectorizer(binary=False, max_features=5)</code>">

Shouldn't this be binary?

</opt>

</choice>

</exercise>

<exercise id="24" title="Text Data True or False">

**True or False**     
*As you increase the value for `max_features` hyperparameter of `CountVectorizer` the training score is likely to go up.*

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

<exercise id="25" title="">

**Instructions:**    
Running a coding exercise for the first time could take a bit of time for everything to load.  Be patient, it could take a few minutes. 

**When you see `____` in a coding exercise, replace it with what you assume to be the correct code.  Run it and see if you obtain the desired output.  Submit your code to validate if you were correct.**

_**Make sure you remove the hash (`#`) symbol in the coding portions of this question.  We have commented them so that the line won't execute and you can test your code after each step.**_


Tasks:     


<codeblock id="">

- Are you ...?

</codeblock>

</exercise>




<exercise id="26" title="What Did We Just Learn?" type="slides, video">
<slides source="module6/module6_end" shot="0" start="12:4510" end="13:2010">
</slides>
---
title: 'Module 1: Introduction and Decision Trees'
description:
  'This chapter will explain the different branches of machine learning and introduce decision trees; a machine learning model used in supervised learning'
prev: chapter0
next: /chapter2
type: chapter
id: 1
---

<exercise id="1" title="Introduction + Supervised vs. Unsupervised learning" type="slides">

<slides source="chapter1_01">
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

<slides source="chapter1_02">
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

<exercise id="10" title="Tabular data and Terminology" type="slides">

<slides source="chapter1_03">
</slides>

</exercise>
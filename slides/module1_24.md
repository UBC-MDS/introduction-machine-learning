---
type: slides
---

# ML model parameters and hyperparameters 

Notes: Script to be added
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>


---

- When you call `fit`, a bunch of values get set, like the split variables and split thresholds. 
- These are called **parameters**
- But even before calling `fit` on a specific data set, we can set some "knobs" that control the learning.
- These are called **hyperparameters**

```python 
df = pd.read_csv('data/cities_USA.csv', index_col=0)
X = df.drop(columns=['vote'])
y = df[['vote']]
df
```

```out 
	      lon	          lat	   vote
1	  -80.162475	  25.692104	 blue
2	  -80.214360	  25.944083	 blue
3	  -80.094133	  26.234314	 blue
4	  -80.248086	  26.291902	 blue
5	  -81.789963	  26.348035	 blue
...	    ...	        ...   	 ...
396	-97.460476	  48.225094	 red
397	-96.551116	  48.591592	 blue
398	-166.519855	 53.887114	 red
399	-163.733617	 67.665859	 red
400	-145.423115	 68.077395	 red
400 rows × 3 columns
```

Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>

---

 In scikit-learn, hyperparameters are set in the constructor:

 ```python 
model = DecisionTreeClassifier(max_depth=3) # this is a "decision stump"
model.fit(X, y)
```

```out 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
```

Here, `max_depth` is a hyperparameter. There are many, many more! See the output above and[here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>

---

```python
dot_data = export_graphviz(model)
graphviz.Source(export_graphviz(model,
                                out_file=None,
                                feature_names=X.columns,
                                class_names=["red", "blue"],
                                impurity=True))
```
```out 


``` 
<img src="module1/largetree.png" alt="This image is in /static" width="50%">


Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>

---

<img src="module1/lat_long.png" alt="This image is in /static" width="50%">

- Let's first calculate the Gini impurity of the full dataset

```python 
gini2(3, 3)
```

```out 
0.5
``` 

Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>

---

- Let's calculate the Gini impurity of the shown split on lon = -97.5
- We now have 2 groups (on either side of red line) so calculate impurity for each group
- We add the results together but weight it by the proportion of observations

```python 
gini2(1, 2)*(3/6) + gini2(2, 1)*(3/6)
```

```out 
0.4444444444444445
```

```python 
gini2(0, 1)*(1/6) + gini2(3, 2)*(5/6)
```

```out 
0.4
```

Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>

---

# Let's practice!

Notes: Script here
<html>
<audio controls >
  <source src="572_placeholder_audio.mp3" />
</audio></html>


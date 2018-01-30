# TensorFlow Classification Example

Previously, we've been generating our own data. Now we will be using real data!

* Pima Indians Diabetes Dataset
* Tf.estimator API
* Categorical and Continuous Features
  * Most won't hve numerical features. Some will have categorical ones
* LinearClassifier and DNNClassifier
  * We will show how to change from linear classifier to DNNClassifiers

The steps include the following:

1. Import our data as a pd DataFrame
2. Normalize our Data



Let's begin with our imports

```py
import pandas as pd
import tensorflow as tf
```

### Step 1: Import our Data 

#### Opening a CSV and setting it to a pandas object

```py
diabetes = pd.read_csv('pima-indians-diabetes.csv')
diabetes.head()
```

```py

Number_pregnant	Glucose_concentration	Blood_pressure	Triceps	Insulin	BMI	Pedigree	Age	Class	Group
0	6	0.743719	0.590164	0.353535	0.000000	0.500745	0.234415	50	1	B
1	1	0.427136	0.540984	0.292929	0.000000	0.396423	0.116567	31	0	C
2	8	0.919598	0.524590	0.000000	0.000000	0.347243	0.253629	32	1	B
3	1	0.447236	0.540984	0.232323	0.111111	0.418778	0.038002	21	0	B
4	0	0.688442	0.327869	0.353535	0.198582	0.642325	0.943638
```

"class" is if they have diabetes or not. The "Group" feature does not mean anything, and was added to teach you how to handle features that are strings. A "categorical string"

First, we must clean our data!

### Step 2: Normalize our data

#### Create a list of columns

```py
diabetes.columns
```

```py
Index(['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree', 'Age', 'Class', 'Group'],
      dtype='object')
```

Now copy everything inside that we want to normalize.   
Items we can't normalize:

* Class: _it's our answer / y._
* Group: _it's a string_
* Age: _Though we technically CAN normalize age, we will be converting it to a categorical column._

```py
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
```

#### Normalize with pandas

We can also normalize with skilearn preprocessing, but here's a little trick with pandas.  
The apply custom lambda expression to take x - the minimum value and divide the x-max - x-min.

```py
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
```

```py
diabetes.head()
```

```py
	Number_pregnant	Glucose_concentration	Blood_pressure	Triceps	Insulin	BMI	Pedigree	Age	Class	Group
0	0.352941	0.743719	0.590164	0.353535	0.000000	0.500745	0.234415	50	1	B
1	0.058824	0.427136	0.540984	0.292929	0.000000	0.396423	0.116567	31	0	C
2	0.470588	0.919598	0.524590	0.000000	0.000000	0.347243	0.253629	32	1	B
3	0.058824	0.447236	0.540984	0.232323	0.111111	0.418778	0.038002	21	0	B
4	0.000000	0.688442	0.327869	0.353535	0.198582	0.642325	0.94363
```

This method is nice since we don't have to import anything.

Now let's make a feature list!




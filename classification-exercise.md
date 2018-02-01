# TensorFlow Classification Exercise

Given some California Census Data, we will predict whether someone makes &gt;$50k or &lt;$50k depending on info. We have numeric columns \(Continuous\) and strings \(Categorical\).

You want to predict the **income\_bracket. **We're going to have to change the string for the y-bracket

Checkout **pandas.apply\(\)**

#### Train split data

categorical columsn vs continuous.

For categoriaal you can use hashbucket or vocab list. \(Use Hashbucket for this one\)

#### Create Input Function

#### Create tf.estimator

Use a Linear Classifier







#### The Data

```py
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

census = pd.read_csv('census_data.csv')
```

If you want to learn whatever the unique values inside a column

```py
census['income_bracket'].unique()
```

```py
array([' <=50K', ' >50K'], dtype=object)
```

We will create a function that will return a 0 if the string is ' &lt;=50K' and 1 if else.

```py
def label_fix(label):
    if (label == ' <=50K'):
        return 0
    return 1
```

Then, we can pass this function into the **pandas.apply\(\)** to run it on all the items in a column.

```py
census['income_bracket'] = census['income_bracket'].apply(label_fix)
census.head()
```

```py

age	workclass	education	education_num	marital_status	occupation	relationship	race	gender	capital_gain	capital_loss	hours_per_week	native_country	income_bracket
0	39	State-gov	Bachelors	13	Never-married	Adm-clerical	Not-in-family	White	Male	2174	0	40	United-States	0
1	50	Self-emp-not-inc	Bachelors	13	Married-civ-spouse	Exec-managerial	Husband	White	Male	0	0	13	United-States	0
2	38	Private	HS-grad	9	Divorced	Handlers-cleaners	Not-in-family	White	Male	0	0	40	United-States	0
3	53	Private	11th	7	Married-civ-spouse	Handlers-cleaners	Husband	Black	Male	0	0	40	United-States	0
4	28	Private	Bachelors	13	Married-civ-spouse	Prof-specialty	Wife	Black	Female	0	0	40	Cuba	0
```

### Perform a Train Test Split on the Data {#Perform-a-Train-Test-Split-on-the-Data}

Create X\_data by making a DataFrame by dropping the y\_label. \(Don't forget the **axis=1**\)

```py
X_data = census.drop('income_bracket', axis=1)
```

Create y\_label

```py
y_labels = census['income_bracket']
```

Now let's create the trainsplit

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.3, random_state=101)
```

## Features

#### Categorical Features

**Take note of categorical vs continuous values!**

```py
census.columns
```

```py
Index(['age', 'workclass', 'education', 'education_num', 'marital_status',
       'occupation', 'relationship', 'race', 'gender', 'capital_gain',
       'capital_loss', 'hours_per_week', 'native_country', 'income_bracket'],
      dtype='object')
```

**Create the tf.feature\_columns for the categorical values. Use vocabulary lists or just use hash buckets.**

Since we know this data has either 'M' or 'F", we can use a vocablist to set our features.

```py
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female', 'Male'])
```

We give the function the column name, and all the possible outcomes.

Now let's do this with a **hash bucket**.

```py
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
```

This will make _up to** **1,000 occupations if needed.  
_Let's continue to do the same with the rest of the categorical columns.

```
workclass, martial_status, relationshi, race, native_country
```

Repeat

```py
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
workc = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
chains = tf.feature_column.categorical_column_with_hash_bucket("martial_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
race = tf.feature_column.categorical_column_with_hash_bucket("race", hash_bucket_size=1000)
home = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)
hrtoy = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
```

#### Continuous Features

For any numerical status, we use continuous features.  
Ours are **education\_num, capital\_gain, capital\_loss, hours\_per\_week**

```py
age = tf.feature_column.numeric_column('age')
edu = tf.feature_column.numeric_column('education_num')
gain = tf.feature_column.numeric_column('capital_gain')
loss = tf.feature_column.numeric_column('capital_loss')
hrs = tf.feature_column.numeric_column('hours_per_week')
```

#### Create a feature list of both of them!

Put them in the same order as your df. Just for it too look nice. 

```py
feat_cols = [age, workc, hrtoy, edu, chains, occupation, relationship, race, gender, gain, loss, hrs, home]
```




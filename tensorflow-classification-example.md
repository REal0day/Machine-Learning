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
from sklearn.model_selection import train_test_split
```

### Step 1: Import our Data

#### Opening a CSV and setting it to a pandas object

```py
diabetes = pd.read_csv('pima-indians-diabetes.csv')
diabetes.head()
```

```py
Number_pregnant    Glucose_concentration    Blood_pressure    Triceps    Insulin    BMI    Pedigree    Age    Class    Group
0    6    0.743719    0.590164    0.353535    0.000000    0.500745    0.234415    50    1    B
1    1    0.427136    0.540984    0.292929    0.000000    0.396423    0.116567    31    0    C
2    8    0.919598    0.524590    0.000000    0.000000    0.347243    0.253629    32    1    B
3    1    0.447236    0.540984    0.232323    0.111111    0.418778    0.038002    21    0    B
4    0    0.688442    0.327869    0.353535    0.198582    0.642325    0.943638
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
    Number_pregnant    Glucose_concentration    Blood_pressure    Triceps    Insulin    BMI    Pedigree    Age    Class    Group
0    0.352941    0.743719    0.590164    0.353535    0.000000    0.500745    0.234415    50    1    B
1    0.058824    0.427136    0.540984    0.292929    0.000000    0.396423    0.116567    31    0    C
2    0.470588    0.919598    0.524590    0.000000    0.000000    0.347243    0.253629    32    1    B
3    0.058824    0.447236    0.540984    0.232323    0.111111    0.418778    0.038002    21    0    B
4    0.000000    0.688442    0.327869    0.353535    0.198582    0.642325    0.94363
```

This method is nice since we don't have to import anything.

Now let's make a feature column.

Make a new variable for each feature.

```py
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
```

These are our continuous values. Now let's work on the non-continuous values. Our categorical values. You can do this by a vocab list or a hash bucket.

### Categorical Values \(non-continuous\)

#### Vocal List

Now, we know that they're only 4 possible 'Groups'. A,B,C, or D.  
What we can do, is while we're defining it, we can say pass in the key, and provide a list of possible categories. That's it!

```py
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
```

#### Hash Bucketing

The important thing to note is that you won't always have a convenient list of categories. For example\) Has all the countries in the world. We can handle this with a hashbucket

```py
assigned_group2 = tf.feature_column.categorical_column_with_hash_bucket('Groups',hash_bucket_size=10)
```

_We won't be using this in this example, ie. we're defining this variable as assigned\_group2._

### Feature Engineering \(Unlocked Special Ability\)

#### Convert a Continuous Column to a Categorical Column.

We have age as a continuous value, but we did not normalize it. By converting a continuous column to a categorical column,  one can get more data out of your data this way.

```
diabetes['Age'].hist(bins=20)
```

![](/assets/Screen Shot 2018-01-29 at 10.43.40 PM.png)

Let's make some boundaries. If we ever want to take a continous value and bucket it into categories

```py
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
```

Now let's make a list of all the feature columns

```py
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]
```

#### Create variable for Train Test

Splice our X data and y-label

```py
X_data = diabetes.drop('Class', axis=1)
X_data.head()
```

```py
    Number_pregnant    Glucose_concentration    Blood_pressure    Triceps    Insulin    BMI    Pedigree    Age    Group
0    0.352941    0.743719    0.590164    0.353535    0.000000    0.500745    0.234415    50    B
1    0.058824    0.427136    0.540984    0.292929    0.000000    0.396423    0.116567    31    C
2    0.470588    0.919598    0.524590    0.000000    0.000000    0.347243    0.253629    32    B
3    0.058824    0.447236    0.540984    0.232323    0.111111    0.418778    0.038002    21    B
4    0.000000    0.688442    0.327869    0.353535    0.198582    0.642325    0.943638    33    C
```

\_Now we have everything, excluding our 'Class' colum\_n

#### Get our y-label

```py
labels = diabetes['Class']
```

**ProTip**: Auto-complete the function and then press \[SHIFT\] + \[TAB\] + \[TAB\] and copy the params.

#### Create variable for Train Dataset![](/assets/Screen Shot 2018-01-29 at 10.54.07 PM.png)

```py
X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.3,random_state=101)
```

We set our test size to 30% and include the same random\_state so we have the same answers.

---

#### Create an input function

```py
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
```

#### Create our model

```py
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
```

```
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmp2y21bt5i
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_session_config': None, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_model_dir': '/var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmp2y21bt5i', '_keep_checkpoint_every_n_hours': 10000, '_tf_random_seed': 1}
```

#### Train Model

```py
model.train(input_fn=input_func, steps=1000)
```

```py
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Saving checkpoints for 1 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmp2y21bt5i/model.ckpt.
INFO:tensorflow:loss = 6.93147, step = 1
INFO:tensorflow:global_step/sec: 137.825
INFO:tensorflow:loss = 3.15373, step = 101 (0.726 sec)
INFO:tensorflow:global_step/sec: 148.378
INFO:tensorflow:loss = 4.95333, step = 201 (0.674 sec)
INFO:tensorflow:global_step/sec: 171.448
INFO:tensorflow:loss = 6.25206, step = 301 (0.587 sec)
INFO:tensorflow:global_step/sec: 158.278
INFO:tensorflow:loss = 4.80319, step = 401 (0.631 sec)
INFO:tensorflow:global_step/sec: 144.448
INFO:tensorflow:loss = 6.62586, step = 501 (0.693 sec)
INFO:tensorflow:global_step/sec: 150.381
INFO:tensorflow:loss = 4.89991, step = 601 (0.663 sec)
INFO:tensorflow:global_step/sec: 162.473
INFO:tensorflow:loss = 4.02916, step = 701 (0.625 sec)
INFO:tensorflow:global_step/sec: 107.217
INFO:tensorflow:loss = 4.81234, step = 801 (0.922 sec)
INFO:tensorflow:global_step/sec: 125.984
INFO:tensorflow:loss = 4.44092, step = 901 (0.794 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmp2y21bt5i/model.ckpt.
INFO:tensorflow:Loss for final step: 6.10011.
Out[53]:
<tensorflow.python.estimator.canned.linear.LinearClassifier at 0x11c880a20>
```

#### Evaluate the inputs

We set shuffle to False so that we compare the same values

```py
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
```

```py
results = model.evaluate(eval_input_func)
```

```py
WARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.
WARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.
INFO:tensorflow:Starting evaluation at 2018-01-30-07:06:29
INFO:tensorflow:Restoring parameters from /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmp2y21bt5i/model.ckpt-1000
INFO:tensorflow:Finished evaluation at 2018-01-30-07:06:31
INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.727273, accuracy_baseline = 0.649351, auc = 0.799712, auc_precision_recall = 0.635924, average_loss = 0.527243, global_step = 1000, label/mean = 0.350649, loss = 5.07471, prediction/mean = 0.348325
```

Now we have our results! The best case is for an _ROC Curve_ of &gt; 0.90 aka 90%. This does depends on what your needs are.

![](/assets/roccomp.jpg)

_I found this photo online to explain what an ROC Curve is._

* .90-1 = excellent \(A\)
* 80-.90 = good \(B\)
* .70-.80 = fair \(C\)
* .60-.70 = poor \(D\)
* .50-.60 = fail \(F\)

Let's see our results!!

```
results
```

```py
{'accuracy': 0.72727275,
 'accuracy_baseline': 0.64935064,
 'auc': 0.799712,
 'auc_precision_recall': 0.63592446,
 'average_loss': 0.52724278,
 'global_step': 1000,
 'label/mean': 0.35064936,
 'loss': 5.0747113,
 'prediction/mean': 0.34832543}
```

Looks like our accuracy is 72.72%.  
Let's get some predictions out of this.

#### Have our Model Create Predictions

```py
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10,num_epochs=1,shuffle=False)
```

```py
predictions = model.predict(pred_input_func)
```

Now predictions is of type generator. So to save our results, let's set it to a variable

```py
my_pred = list(predictions)
my_pred
```

```py
[{'class_ids': array([1]),
  'classes': array([b'1'], dtype=object),
  'logistic': array([ 0.51357734], dtype=float32),
  'logits': array([ 0.05432272], dtype=float32),
  'probabilities': array([ 0.48642266,  0.51357734], dtype=float32)},
 {'class_ids': array([1]),
  'classes': array([b'1'], dtype=object),
  'logistic': array([ 0.59563601], dtype=float32),
  'logits': array([ 0.38731456], dtype=float32),
  'probabilities': array([ 0.4043639 ,  0.59563601], dtype=float32)},
 {'class_ids': array([0]),
  'classes': array([b'0'], dtype=object),
  'logistic': array([ 0.3654128], dtype=float32),
  'logits': array([-0.55194706], dtype=float32),
  'probabilities': array([ 0.63458717,  0.36541277], dtype=float32)}
  ---snip---
```

As you can see with the first item in this dictionary, the class\_ids has the array\(1\). This means the model has classified the person of having diabetes.

```py
class_ids': array([1]),
  'classes': array([b'1'], dtype=object)
```

We can see the probability that the person has diabetes here:

```py
---snip---
'probabilities': array([ 0.48642266,  0.51357734], dtype=float32)
```

### Dense Neural Network \(DNN\) Classifier

#### Create our DNN Model

Sets how many layers you want. You set a nueral per layer. \[10,10,10\] == 10 neurons in 3 layers, and they're all connected to each other.

```py
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)
```

#### Create our embedded group column

We must put our columns in an embedded column. Set dimension to 4, as we have 4 classifications. "A", "B", "C", "D".

```py
embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
```

#### Create our features column

```py
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_column, age_buckets]
```

_Just like we did before, but we change the **assigned\_group** with the **embedded\_group **_

#### Create our Input Function

```py
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)
```

#### Create our DNN Model

```py
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols, n_classes=2)
```

```py
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmpr4am2vdj
INFO:tensorflow:Using config: {'_tf_random_seed': 1, '_session_config': None, '_save_checkpoints_steps': None, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_keep_checkpoint_every_n_hours': 10000, '_keep_checkpoint_max': 5, '_model_dir': '/var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmpr4am2vdj'}
```

#### Train our DNN Model

```py
dnn_model.train(input_fn=input_func,steps=1000)
```

```py
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Saving checkpoints for 1 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmpr4am2vdj/model.ckpt.
INFO:tensorflow:step = 1, loss = 13.0735
INFO:tensorflow:global_step/sec: 190.77
INFO:tensorflow:step = 101, loss = 5.25569 (0.526 sec)
INFO:tensorflow:global_step/sec: 225.397
---snip---
INFO:tensorflow:step = 901, loss = 8.17106 (0.406 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmpr4am2vdj/model.ckpt.
INFO:tensorflow:Loss for final step: 3.94409.
Out[50]:
<tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x1092c6b70>
```

#### Create our Evaluation function to test our model

```py
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
```

Now let's test it!

#### Evaluated our Model with eval data

Again, this will take our X\_data, and check to see if it can predict the output y-label.

```py
dnn_model.evaluate(eval_input_func)
```

```py
WARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.
WARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.
INFO:tensorflow:Starting evaluation at 2018-01-31-00:07:29
INFO:tensorflow:Restoring parameters from /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmpr4am2vdj/model.ckpt-1000
INFO:tensorflow:Finished evaluation at 2018-01-31-00:07:31
INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.728346, accuracy_baseline = 0.65748, auc = 0.790144, auc_precision_recall = 0.63273, average_loss = 0.520272, global_step = 1000, label/mean = 0.34252, loss = 5.08265, prediction/mean = 0.336591-
---snip---
{'accuracy': 0.72834647,
 'accuracy_baseline': 0.65748036,
 'auc': 0.79014391,
 'auc_precision_recall': 0.63273042,
 'average_loss': 0.5202716,
 'global_step': 1000,
 'label/mean': 0.34251967,
 'loss': 5.0826535,
 'prediction/mean': 0.33659098}
```

We can try to increase the neurons in the hidden layers to see if we can get a better level of accuracy.

```py
dnn_model = tf.estimator.DNNClassifier(hidden_units=[20,20,20],feature_columns=feat_cols,n_classes=2)
```

#### Train Time

```py
dnn_model.train(input_fn=input_func,steps=1000)
```

#### Eval Time

```py
dnn_model.evaluate(eval_input_func)
```

```py
{'accuracy': 0.70866144,
 'accuracy_baseline': 0.65748036,
 'auc': 0.79358524,
 'auc_precision_recall': 0.62722367,
 'average_loss': 0.53591853,
 'global_step': 1000,
 'label/mean': 0.34251967,
 'loss': 5.2355118,
 'prediction/mean': 0.38641998}
```

...wat? Our results indicate it was worse. I'm going to increase the epoch to see if that helps. Maybe it didn't have enough time to get a good level of accuracy, given the number of neurons.

```py
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=2500,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[20,20,20],feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
dnn_model.evaluate(eval_input_func)
```

```py
{'accuracy': 0.74409449,
 'accuracy_baseline': 0.65748036,
 'auc': 0.79688907,
 'auc_precision_recall': 0.64370602,
 'average_loss': 0.52519315,
 'global_step': 1000,
 'label/mean': 0.34251967,
 'loss': 5.130733,
 'prediction/mean': 0.31303018}
```

Alright so much better! We increased our accuracy by 2% from our highest point!

\(epoch=1000, DNN\[10,10,10\], 72.83%\)  
\(epoch=1000, DNN\[15,15,15\], 70.87%\)  
\(epoch=2000, DNN\[10,10,10\], 74.41%\)


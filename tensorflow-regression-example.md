# TensorFlow Regression Example

## Part I

We will now use a more realistic regression example and introduce **tf.estimator**.

It will take place in the following steps:

1. Collect Data
2. Create Variables
3. Create Placeholders
4. Define operations in your Graph \(Set operations being taken\)
5. Define error or loss function
6. setup trainer
7. Initialize global objects
8. If big dataset:
   1. Create batches

Let's start with our imports

```py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
```

Sweet. Let's create a "HUGE" dataset_. Huge is relative. This will have 1M datapoints._

```py
x_data = np.linspace(0.0,10.0,1000000)
```

Now let's create the same amount of noise as our dataset. \(1M\)

```py
noise = np.random.randn(len(x_data))
```

Our line will be modeled as y = mx + b. We will set b = 5 and add noise.

```py
y_true = (0.5 * x_data) + 5 + noise
```

_The noise is added so it's our true line is not just a perfect fitted line. The 5 represents the shift in b that we want._

Let's use pandas to concatenate our data.

**Create our x\_dataFrame**

```py
x_df = pd.DataFrame(data=x_data, columns=['X Data'])
```

**Create our y\_dataFrame**

```py
y_df = pd.DataFrame(data=y_true, columns=['Y'])
```

Now let's check the first five entries of x\_df

```py
x_df.head()
```

```py
     X Data
0    0.00000
1    0.00001
2    0.00002
3    0.00003
4    0.00004
```

Now let's **concat** them both to have one frame!

```
my_data = pd.concat([x_df,y_df], axis=1)
```

Output of my\_data.head\(\)

```py
     X Data     Y
0    0.00000    5.718261
1    0.00001    5.000671
2    0.00002    5.544956
3    0.00003    5.070396
4    0.00004    4.691148
```

_We add the axis=1 so that the data isn't stacked like a pancake!_

Now let's say we want to graph this. Unfortunately, graphing a million plot points will take awhile.  
Let's only graph a small random sample of 10.

```py
my_data.sample(n=10)
```

Output

```py
index    X Data        Y
140413    1.404131    5.637388
763166    7.631668    9.500907
210459    2.104592    5.871338
238403    2.384032    6.298123
74718    0.747181    5.466532
68900    0.689001    4.138463
691081    6.910817    9.494513
797712    7.977128    7.670427
517287    5.172875    6.555583
63754    0.637541    6.250593
```

**Create a scatter plot w/ 250 values**

```py
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
```

![](/assets/cp1.png)

Now let's use TensorFlow to train this model. Now, we can't run 1M of points at a time, we have to create **batches** of data. They're no true right or wrong answer for batch\_sizes. It depends on your data.

```py
batch_size = 8
```

**Create our slope and b variable              
**_They're random numbers._

```
m = tf.Variable(0.3)
b = tf.Variable(0.11)
```

**Create our placeholders**. 1 for x and 1 for y.  
Don't forget to set the data type and the size of the batch.

```py
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])
```

_Remember yph is the _**True**_ answer._

**Define our model**

```py
y_model = m*xph + b
```

Great! Now let's **Create our error**.

```py
error = tf.reduce_sum(tf.square(yph - y_model))
```

Now for our **Gradient Descent Optimizer** and create a train variable using our optimizer on error.=

```py
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
```

Almost done! Just need to initialize our global variables, then run our analysis!

```py
init = tf.global_variables_initializer()
```

```py
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {xph:x_data[rand_ind], yph:y_true[rand_int]}
        sess.run(train, feed_dict = feed)
    model_m, model_b = sess.run([m,b])
```

Alright, so what's happening here?  
We're grabbing 8 random data points. **rand\_ind** grabs a random index of our data, and then put that as our xph and yph, which becomes our **feed**. Feed are two points \(x,y-true\) and it is added to the dictionary **feed{}**. We then use train to adjust errors, and set our feed-dict to begin to run our analysis. Runt his. If it takes too long, lower the number of batches.

```py
model_m
```

```py
0.52406013
```

```py
model_b
```

```py
4.9413366
```

Set v\_hat and graph!

```py
y_hat = x_data*model_m + model_b
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data,y_hat, 'r')
```

![](/assets/plot.png)

## Part II - **Estimator API**

Now we will solve the regression task using the **Estimator API**.

They're are lot of other higher level APIs \(Keras, Layers, etc.\) but we will conver those later on in the _Miscellaneous Section._

The **tf.estimator** has many different options/types

* tf.estimator.**LinearClassifier**
  * Constructs a linear classification model
* tf.estimator.**LinearRegressor**
  * Constructs a linear regression model
* tf.estimator.**DNNClassifer**
  * Construct a neural network classification model
* tf.estimator.**DNNRegressor**
  * Constructs a neural network regression model
* tf.estimator.**DNNLinearCombinedClassifer**
  * Constructs a neural network and linear combined classification
* tf.estimator.**DNNLinearCombinedRegressor**
  * Constructs a neural network and linear combined regression model

To use the Estimator API:

1. Define a list of feature columns
2. Create the Estimator Model
3. Create a Data Input Function
4. Call train\(\), eval\(\), and predict\(\) on the estimator object.

### **Creating the feature columns list for the Estimator**

```py
feat_cols = [ tf.feature_column.numeric_column('x', shape=[1]) ]
```

Now we setup our estimator. This is the main part of the API. We will do a LinearRegressor and point to the feature columns. We will see more complex examples with multiple featuresl.

```py
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
```

_There will be an output, but it's just default configuration stuff._

### Create training and evaluation variables \(70%,30%\)

We are splitting up the data into a training set and an evaluation set. We set the test\_size to 0.3 aka 30% of an evaluation size and 70% of a test size.

```py
from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
```

Let's see if we got what we wanted

```py
print(x_train.shape)
print(x_eval.shape)
```

```py
(700000,)
(300000,)
```

70% of 1M is 700,000, so it has appeared to work.

### Setup a Estimator Inputs

You need to have a n input function that kinda acts like your feed dictionary and batch_size_ indicator all at once. We will be inputing from an nump array.** You can also send in pandas array! **We then define a dictionary of **'x' key** to the values of **xtrain**,  then **y\_train** as the

```py
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
```

Let's copy and paste this to get 2 more variables, **train\_input\_func, **and ** eval\_input\_func.**

```py
train_input_func = input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
eval_input_func = input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
```

### Train the Estimator

Time to train this bitch. Let's give it 1000 steps and see how it does.

```py
estimator.train(input_fn=input_func, steps=1000)
```

```py
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Saving checkpoints for 1 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmppdgsysdb/model.ckpt.
INFO:tensorflow:loss = 543.0, step = 1
INFO:tensorflow:global_step/sec: 563.558
INFO:tensorflow:loss = 26.5623, step = 101 (0.178 sec)
INFO:tensorflow:global_step/sec: 572.734
INFO:tensorflow:loss = 49.0421, step = 201 (0.174 sec)
INFO:tensorflow:global_step/sec: 476.958
INFO:tensorflow:loss = 22.2036, step = 301 (0.210 sec)
INFO:tensorflow:global_step/sec: 556.155
INFO:tensorflow:loss = 13.5479, step = 401 (0.181 sec)
INFO:tensorflow:global_step/sec: 470.728
INFO:tensorflow:loss = 3.91283, step = 501 (0.212 sec)
INFO:tensorflow:global_step/sec: 594.255
INFO:tensorflow:loss = 12.6644, step = 601 (0.170 sec)
INFO:tensorflow:global_step/sec: 445.313
INFO:tensorflow:loss = 7.96772, step = 701 (0.224 sec)
INFO:tensorflow:global_step/sec: 440.612
INFO:tensorflow:loss = 19.8273, step = 801 (0.227 sec)
INFO:tensorflow:global_step/sec: 428.152
INFO:tensorflow:loss = 9.37158, step = 901 (0.236 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmppdgsysdb/model.ckpt.
INFO:tensorflow:Loss for final step: 12.2294.
```

### Evaluate the train\_metric with the Estimator

We use the **train\_input\_func** for the input\_fn because we do not want the data to be shuffled. Recall we turned shuffle off.

```py
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
```

```py
INFO:tensorflow:Starting evaluation at 2018-01-29-00:00:22
INFO:tensorflow:Restoring parameters from /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmppdgsysdb/model.ckpt-1000
INFO:tensorflow:Evaluation [1/1000]
INFO:tensorflow:Evaluation [2/1000]
INFO:tensorflow:Evaluation [3/1000]
---snip---
INFO:tensorflow:Evaluation [997/1000]
INFO:tensorflow:Evaluation [998/1000]
INFO:tensorflow:Evaluation [999/1000]
INFO:tensorflow:Evaluation [1000/1000]
INFO:tensorflow:Finished evaluation at 2018-01-29-00:00:27
INFO:tensorflow:Saving dict for global step 1000: average_loss = 1.0769, global_step = 1000, loss = 8.61518
```

### Evaluate the eval\_metrics with the Estimator

```py
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
```

```py
INFO:tensorflow:Starting evaluation at 2018-01-29-00:01:29
INFO:tensorflow:Restoring parameters from /var/folders/3v/vpv_7q_95dj_87nc7vkrf88h0000gn/T/tmppdgsysdb/model.ckpt-1000
INFO:tensorflow:Evaluation [1/1000]
INFO:tensorflow:Evaluation [2/1000]
INFO:tensorflow:Evaluation [3/1000]
---snip---
INFO:tensorflow:Evaluation [997/1000]
INFO:tensorflow:Evaluation [998/1000]
INFO:tensorflow:Evaluation [999/1000]
INFO:tensorflow:Evaluation [1000/1000]
INFO:tensorflow:Finished evaluation at 2018-01-29-00:01:34
INFO:tensorflow:Saving dict for global step 1000: average_loss = 1.06595, global_step = 1000, loss = 8.52763
```

Now let's print both metrics and see if they're close. this will let us know if we are overfitting to our data.A good indicator is when you havea realy low loss, but high on th eval data. We want them to be close to each other. Preferably not e these two metrics to be as cval perfroms worse than train\_set. **If training metrics is way higher, but lower. than eval, then you're overfitting.**

```py
print('TRAINING DATA METRICS')
print(train_metrics)
```

```py
TRAINING DATA METRICS
{'average_loss': 1.0768981, 'global_step': 1000, 'loss': 8.6151848}
```

```py
print('EVAL METRICS')
print(eval_metrics)
```

```py
EVAL METRICS
{'average_loss': 1.0659537, 'global_step': 1000, 'loss': 8.5276299}
```




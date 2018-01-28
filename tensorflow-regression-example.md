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




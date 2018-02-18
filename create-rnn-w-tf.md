# Create RNN w/ TF

#### Imports

```py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
```

Create a class that allows us to initialize the data and send batches back.

```py
class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points # how fine of a resolution are we creating
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sing(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # Grab a random starting point for each batch of data
        rand_start = np.random.rand(batch_size, 1)

        # Convert to be on the same time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

        # Create batch series on the x-axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution

        # Create the y data for the time series x-axis from previous step
        y_batch = np.sin(batch_ts)

        # FORMATTING for RNN.
        if (return_batch_ts):
            return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1) , batch_ts
        return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1)
```

So far, we have a timeseries class that takes in the number of points wanted, and the xmin and xmax. It then creates a bunch of attributes to store information. It creates the resolution , x\_data, and y\_true. y\_true takes in x\_data through the numpy sin function.

We will also create a convience method called **return true.** it takes in any series of x values and will return np.This will makes things easier.

Create a function to generate batches of this data

Let's go over the **next\_batch\(\).**

1. We created a random starting point. However we don't know if it's on the time-series data or not because the time series data was first defined when we initialzed this, with a xmin, xmax, and a number of points. 
2. We then convert this random start to BE on the tiem series. We do that by multiply the random start with the \(xmax - xmin - \(steps \* resolution\).
3. We need to create the X\_data for our time series batch. It's our starting point that we decided on + \(0, steps + 1\) and multiply it by the resolution. It's that resoluition times the arrangement, then t-start plus the following points.
4. We take the np.sin of our batch, to determine our true value.

Let's create some data so we can visualize what's happening.

```py
# TimeSeriesData(num_points, xmin, xmax)
ts_data = TimeSeriesData(250,0,10)
```

```
250 points between points 0 and 10. Now let's plot it.
```

```py
plt.plot(ts_data.x_data, ts_data.y_true)
```

![](/assets/Screen Shot 2018-02-02 at 10.16.10 PM.png)

Of **ts\_data**, we have our **x\_data** and our **y\_data**

#### Create Random 30-step batches

We can use this for predictions steps in the future.

```py
num_time_steps = 30
```

#### Create 1 batch of 30 steps

```py
y1,y2,ts = ts_data.next_batch(1,num_time_steps, True)
```

True says we want to True to say we want that time series data so we can plot it.  
But before we can do that, we have to **flatten** it, meaning we need to put it in a 1-D array.

```py
ts
```

```py
array([[ 5.25343895,  5.29343895,  5.33343895,  5.37343895,  5.41343895,
         5.45343895,  5.49343895,  5.53343895,  5.57343895,  5.61343895,
         5.65343895,  5.69343895,  5.73343895,  5.77343895,  5.81343895,
         5.85343895,  5.89343895,  5.93343895,  5.97343895,  6.01343895,
         6.05343895,  6.09343895,  6.13343895,  6.17343895,  6.21343895,
         6.25343895,  6.29343895,  6.33343895,  6.37343895,  6.41343895,
         6.45343895]])
```

Let's check this matrix

```py
ts.shape
```

```py
(1, 31)
```

We need to make it into a vector

```py
ts.flatten().shape # this is what matplotlib needs
```

```py
(31,)
```

Perfect. Now we must set the dimensions so they match up. As far as the training batches, 1 is shifted over. so we must start our training set data +1.

```py
y1,y2,ts = ts_data.next_batch(1,num_time_steps, True)
plt.plot(ts.flatten()[1:], y2.flatten(), '*')
```

![](/assets/Screen Shot 2018-02-02 at 10.27.06 PM.png)

This is our batch of data in our dataset. Let's see it relative to our entire data.

```py
plt.plot(ts_data.x_data, ts_data.y_true, label='sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '+')
plt.legend()
plt.tight_layout()
```

![](/assets/Screen Shot 2018-02-02 at 10.28.36 PM.png)

So we are trying to predict a time-series shifted by 1 step.

### Create Training Data

```py
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)
```

This training set is a bunch of values

```py
train_inst
```

```py
array([ 5.        ,  5.04133333,  5.08266667,  5.124     ,  5.16533333,
        5.20666667,  5.248     ,  5.28933333,  5.33066667,  5.372     ,
        5.41333333,  5.45466667,  5.496     ,  5.53733333,  5.57866667,
        5.62      ,  5.66133333,  5.70266667,  5.744     ,  5.78533333,
        5.82666667,  5.868     ,  5.90933333,  5.95066667,  5.992     ,
        6.03333333,  6.07466667,  6.116     ,  6.15733333,  6.19866667,
        6.24      ])
```

Let's plot where our training data SHOULD predict to.

```py
plt.title('A Training Instance')
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label='Instance')

plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=7, label='Target')
```

![](/assets/Screen Shot 2018-02-02 at 10.44.05 PM.png)

Given the blue points, can you predict the black points?

---

## Create the Model {#Generating-New-Sequences}

```py
tf.reset_default_graph()
```

#### Constraints {#Generating-New-Sequences}

```py
# Just one feature, the time series
num_inputs = 1

# 100 neuron layer, play with this
num_neurons = 100

# Just one output, predicted time series
num_outputs = 1

# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.0001

# how many iterations to go through (training steps), you can play with this
num_train_iterations = 2000

# Size of the batch of data
batch_size = 1
```

#### Placeholders {#Generating-New-Sequences}

```py
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
```

### RNN Cell Layer {#RNN-Cell-Layer}

#### Create Cell

```py
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)

# n_neurons = 100
# n_layers = 3

# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)

# n_neurons = 100
# n_layers = 3

# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
#           for layer in range(n_layers)])
```

## Dynamic RNN Cell {#Generating-New-Sequences}

```py
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```

### Loss Function and Optimizer {#Loss-Function-and-Optimizer}

```py
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
```

#### Init Variables {#Init-Variables}

```py
init = tf.global_variables_initializer()
```

#### Create Session {#Generating-New-Sequences}

```py
# ONLY FOR GPU USERS:
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
```

You can save your model by doing the following

```py
saver = tf.train.Saver()
```

Session!

```py
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):

        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:

            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    # Save Model for Later
    saver.save(sess, "./rnn_time_series_model")
```

```py
0     MSE: 0.418418
100     MSE: 0.0503464
200     MSE: 0.0509452
300     MSE: 0.00751106
400     MSE: 0.0235597
500     MSE: 0.0229088
600     MSE: 0.00144745
700     MSE: 0.0304383
800     MSE: 0.00336185
900     MSE: 0.00579273
1000     MSE: 0.00723241
1100     MSE: 0.00245652
1200     MSE: 0.00164544
1300     MSE: 0.00314354
1400     MSE: 0.00105241
1500     MSE: 0.000656848
1600     MSE: 0.00864561
1700     MSE: 0.00708487
1800     MSE: 0.00842478
1900     MSE: 0.000408624
```

### Predicting a time series t+1 {#Predicting-a-time-series-t+1}

#### Load saved Model {#Generating-New-Sequences}

```py
with tf.Session() as sess:                          
    saver.restore(sess, "./rnn_time_series_model")   

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
```

Plot!

```py
plt.title("Testing Model")

# Training Instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")

# Target to Predict
plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")

# Models Prediction
plt.plot(train_inst[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")

plt.xlabel("Time")
plt.legend()
plt.tight_layout()
```

![](/assets/download-2.png)

---

# Generating New Sequences {#Generating-New-Sequences}

After that, we will give it a seed series, and ask it to predict a new sequence.Here, we feed it zeros. It's passing this off of what it knows about the sin\(x\). We

```py
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # SEED WITH ZEROS
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])
```

We first restore our model. Create 30 zero sequence seed, or the num\_time\_steps.

**X\_batch** = create new batch. pass in zero sequence array, going backward to the amount of the number of steps. and then we reshape it for the RNN. _        
_**y\_batch **= run the output, passing in the X:X\_batch we created.  
**zero\_seq\_seed **= append the new values to the zero\_seq\_seed At the ery end, we should ahve 30 zeros and then the generated values.

Now time to plot.

```py
plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
```

![](/assets/download.png)

```py
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # SEED WITH Training Instance
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(training_instance) -num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        training_instance.append(y_pred[0, -1, 0])
```

Plot

```py
plt.plot(ts_data.x_data, ts_data.y_true, "b-")
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps], "r-", linewidth=3)
plt.xlabel("Time")
```

![](/assets/download-1.png)


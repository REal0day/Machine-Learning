# TensorFlow - Create a Neural Network Pt.2

We will create a very simple example neural network.

![](/assets/Screen Shot 2018-01-23 at 5.57.30 PM.png)

Let's set our constant variables such as the number of features and the number of neurons.

```py
import numpy as np
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)

n_features = 10
n_desn_neurons = 3
```

Great! Now let's create a placeholder for "x" aka our inputs.

```py
x = tf.placeholder(tf.float32,(None,n_features))
```

_Since we know the number of features, we can set that to our variable. But since we don't know how many x inputs we will have, let's use None._

Now to add weights. We will use **random\_normal**, a TensorFlow module for assigning weight amounts. We will go into more details about how to decide what weights and bias to use.

```py
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
```

For our bias term, we can have it be 0's or 1's. The shape will be the number of neruons. The reason for this is because w is \* by x. **b must match the number of neurons.**

```py
b = tf.Variable(tf.ones([n_dense_neurons]))
```

Setting our variables

```py
xW = tf.matmul(x,W)
z = tf.add(xW,b)
```

We will turn a into a **sigmoid\(\)**. Again, sigmoid\(\) is any number from zero to one. \[0,1\]

![](/assets/im3port.png)

**a** \(our neuron's prediction\)

```py
a = tf.nn.sigmoid(z)
```

Great! Now let's initialize our global variables...

```py
init = tf.global_variables_initializer()
```

...and run this!

```py
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})
```

If we print\(layer\_out\), we get the following: \(_Your numbers will vary_\)

```c
[[ 0.31433171  0.48861519  0.23742266]]
```

---

## Simple Regression Example

Notice that we are not adjusting the values of **W **or **b**. That is not how you run a neural network. We need to add a cost function and to add an optimizer. We will add a regression.

**Create our x\_data \(input data\)**

```py
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
```

This looks like the following: \(_your data will be different_\)

```py
array([ 0.5420333 ,  1.17575569,  0.85241231,  2.50514314,  4.67005971,
        4.41685654,  6.66701681,  6.69180648,  7.54731409,  9.03483077])
```

Now let's **Create our y\_label** \(_expected results_\)

```py
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
```

Output

```py
array([ 0.39254011,  0.74493691,  0.76194526,  4.35999157,  4.09925768,
        5.71055309,  7.29828109,  8.30361433,  9.47258288,  9.5273703 ])
```

Awesome! Let's plot our points. We will use 'g' for green and '\*' for \*s. 

```py
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(x_data,y_label, 'g*')
```

![](/assets/NN1.png)

Now we are trying to solve y=mx = b.  
So we will initilize these with two random values. Something close to what we want, but it will become closer as we train our model more. It will fix it with the cost function and optimizer.

```py
m = tf.Variable(0.44)
b = tf.Variable(0,87)
```

---

### Cost Function

We will set the error to 0, and it will increase as the difference between y and y\_hat. Another thing to notice is that we type cast our variables to float32 so they work.

```py
error = 0

m = tf.cast(m, tf.float32)
x = tf.cast(m, tf.float32)
b = tf.cast(m, tf.float32)

for x,y in zip(x_data,y_label):
    
    y_hat = m*x + b
    
    error += (y-y_hat)**2
```

Perfect! Now let's set our optimizer and a learning rate of 0.001. Remember that if the learning rate is too high, it will not get leave the parabola. If it's too low, it'll take a lot longer.![](/assets/Learning Rate2.png)![](/assets/learning rate.png)

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
```

Now we must tell our model what we want to optimize. In this case, it's our error.

```
train = optimizer.minimize(error)
```

Perfect! Now we just need to intilize the global variables and run our model! Let's start with letting it train once.

```py
with tf.Session() as sess:
    
    sess.run(init)
    training_steps = 1
    
    for i in range(training_steps):
        sess.run(train)
    
    final_slope , final_intercept = sess.run([m,b])
```

Let's graph it!

```py
# With 1 training steps
x_test = np.linspace(-1,11,10)

# y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot)
plt.plot(x_data,y_label,'*r')
```

![](/assets/onetimetrain.png)

As we can see it looks pretty damn good. But let's see if we can't make it better by training it a bit more. Maybe 99 more times. Let's re-run our session, and graph it.

```py
with tf.Session() as sess:    
    sess.run(init)
    training_steps = 100
    
    for i in range(training_steps):    
        sess.run(train)
    
    final_slope , final_intercept = sess.run([m,b])

# With 100 training steps
x_test = np.linspace(-1,11,10)

# y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot)
plt.plot(x_data,y_label,'*r')
```

![](/assets/100train.png)

That's better! Maybe it's hard to see the difference though. Let me create a graph with both the lines on it.

![](/assets/Both.png)

_The green is the line that had been trained 100x and the blue line had been trained once._




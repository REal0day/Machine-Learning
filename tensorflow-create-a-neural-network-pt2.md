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




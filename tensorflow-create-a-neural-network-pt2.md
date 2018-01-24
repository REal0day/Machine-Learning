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

We will turn a into a **sigmoid\(\)**. Again, a sigmoid\(\) is any number from zero to one. \[0,1\]

![](/assets/im3port.png)

```py
a = tf.nn.sigmoid(z)
```




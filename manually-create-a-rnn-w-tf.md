# Manually Create a RNN w/ TF

We will create a 3 Neuron Recurrent Neural Network with TensorFlow. The main idea to focus on here is the input format of the data.![](/assets/Screen Shot 2018-02-01 at 3.09.41 AM.png)

Let's start by running the RNN for 2 batches of data, t=0 and t=1.

Each Recurrent Neruon has 2 sets of weights:

* Wx for input weights on X
* Wy for weights on output of original X

![](/assets/Screen Shot 2018-02-01 at 3.20.19 AM.png)  
**num\_batches**: size of one sample.  
**batch\_size: **Samples of dataset  
**time\_steps**: Intervals per Sample

Feed in based on the timestamp. from t=0 --&gt; t=4

Steps:

1. Create Constants
2. Create Placeholders
3. Create Nuerons
4. Create Bias Terms
5. Graph/cast our output as tanh\(\)
6. Initialize Variables
7. Create Data
   1. Timestamps
8. Run session



#### Imports

```py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
```

#### 1. Create Constants

```py
num_inputs = 2
num_neurons =3
```

#### 2. Create Placeholders \(for each timestamp\)

```py
x0 = tf.placeholder(tf.float32,[None,num_inputs])
x1 = tf.placeholder(tf.float32,[None,num_inputs])
```

_We're only doing two but for a real RNN, you . would do 1 per timestamp_

#### 3. Create Neurons

```py
Wx = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
Wy = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))
```

#### 4. Create Bias Terms

```py
b = tf.Variable(tf.zeros([1,num_neurons]))
```

#### 5. Graph/cast our output as tanh\(\)

y0 is our original output, multipled by the first weights plus the bias term, then pass it to an activation function.

y1 is the NEXT timestep , we take that output, and multiiple and add the current input plus the bias

```py
y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)
```

#### 6. Initialize Variables

```py
init = tf.global_variables_initializer()
```

#### 7. Create Data

```py
# Timestamp 0
x0_batch = np.array([ [0,1], [2,3], [4,5] ])

# Timestamp 1
x1_batch = np.array([ [100,101], [102,103], [104,105] ])
```

#### 8. Run Session

```py
# Run Session
with tf.Session() as sess:
    sess.run(init) #initilze variables
    
    y0_output_vals, y1_output_vals = sess.run([y0,y1], feed_dict={x0:x0_batch, x1:x1_batch}) 
```




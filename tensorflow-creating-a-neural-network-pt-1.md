# TensorFlow - Create a Neural Network Pt.1

Let's create a simple linear fit to some 2-D Data.

The steps are the following:

1. Build a Graph
2. Initiate the Session
3. Feed Data IN and get Output

Our graph will look like the following:![](/assets/Screen Shot 2018-01-23 at 5.57.30 PM.png)Afterwards, we will add the cost function so we can train our network to optimize the parameters. Let's get started!

---

**Import and set our seeds**

```py
import numpy as np
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)
```

**Create a Random 5x5 Matrix**

```py
rand_a = np.random.uniform(0,100,(5,5))
rand_a
```

Output

```py
array([[ 51.63986277,  57.06675869,   2.84742265,  17.15216562,
         68.52769817],
       [ 83.38968626,  30.69662197,  89.36130797,  72.15438618,
         18.99389542],
       [ 55.42275911,  35.2131954 ,  18.18924027,  78.56017619,
         96.54832224],
       [ 23.23536618,   8.35614337,  60.35484223,  72.89927573,
         27.62388285],
       [ 68.53063288,  51.78674742,   4.84845374,  13.78692376,
         18.69674261]])
```

And let's create one more.

```py
rand_b = np.random.uniform(0,100,(5,1))
rand_b
```

Output

```py
array([[ 99.43179012],
       [ 52.06653967],
       [ 57.87895355],
       [ 73.48190583],
       [ 54.19617722]])
```

Sweet! Looking good!

---

**Create Placeholders**

```py
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
```

TensorFlow can understand the built-in python operators.  

```py
add_op = a + b
mul_op = a * b
```

Let's run it  
We must add an operation and a feed dictionary.

```py
with tf.Session() as sess:
    
    add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)
```

Output

```py
[[ 151.07165527  156.49855042  102.27921295  116.58396149  167.95948792]
 [ 135.45622253   82.76316071  141.42784119  124.22093201   71.06043243]
 [ 113.30171204   93.09214783   76.06819153  136.43911743  154.42727661]
 [  96.7172699    81.83804321  133.83674622  146.38117981  101.10578918]
 [ 122.72680664  105.98292542   59.04463196   67.98310089   72.89292145]]
```

Sweet! Now let's run the multiplication operator.

```py
with tf.Session() as sess:
    
    add_result = sess.run((mul_op), feed_dict={a:rand_a, b:rand_b})
    print(add_result)
```

Output

```py
[[ 5134.64404297  5674.25         283.12432861  1705.47070312
   6813.83154297]
 [ 4341.8125      1598.26696777  4652.73388672  3756.8293457    988.9463501 ]
 [ 3207.8112793   2038.10290527  1052.77416992  4546.98046875
   5588.11572266]
 [ 1707.37902832   614.02526855  4434.98876953  5356.77734375
   2029.85546875]
 [ 3714.09838867  2806.64379883   262.76763916   747.19854736
   1013.29199219]]
```




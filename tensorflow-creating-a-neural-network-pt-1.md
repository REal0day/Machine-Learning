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




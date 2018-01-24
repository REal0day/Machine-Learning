# TensorFlow Basic Syntax

**Import TensorFlow and Check Version**

```py
import tensorflow as tf
print(tf.__version__)

1.3.0
```

The word **tensor** is n-dimensional array.  
Let's create one!

**Create Tensor Object**

```py
hello = tf.constant("Hello")
world = tf.constant("World")
type(hello)

tensorflow.python.framework.ops.Tensor
```

**Run a TensorFlow Session**

```py
with tf.Session() as sess:
    result = sess.run(hello+world)
print(result)

b'Helloworld'
```

The 'b' represents a "bytes-literal"." If you're familiar with Java or C\#, think of str as String and bytes as byte\[ \].

**Create Tensor Constant**

```py
a = tf.constant(10)
b = tf.constant(20)
type(a)

tensorflow.python.framework.ops.Tensor
```

TensorFlow also keeps track of the amount of times an operation is done.

```py
a + b

<tf.Tensor 'add_1:0' shape=() dtype=int32>
```

This is our first time doing this operation, as you can see by 'add\_1:0'. Let's try it again.

```py
a + b

<tf.Tensor 'add_2:0' shape=() dtype=int32>
```

Now it says 'add\_2:0'.

**Add two variables in TensorFlow Session**

```py
with tf.Session() as sess:
    result = sess.run(a+b)
print(result)

30
```

**Create Matrices**

```py
const = tf.constant(10)
fill_mat = tf.fill((4,4), 10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4), mean=0, stddev=1.0)
myrandu = tf.random_uniform((4,4), minval=0, maxval=1)
```

First param is the size of the matrix, they're more specifications one can create as well. 

**Create a list of "operations"**

```
my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]
```

**Now let's run them in an interactive session. It works in a notebook/interactive sessions. Otherwise, most the time one will use: with tf.Session\(\) as sess: .**

**Create an Interactive Sessions**

```py
sess = tf.InteractiveSession()

for op in my_ops:
    print(sess.run(op))
    
10
[[10 10 10 10]
 [10 10 10 10]
 [10 10 10 10]
 [10 10 10 10]]
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
[[ -1.41266870e+00   4.48324502e-01   7.23314226e-01   1.27894032e+00]
 [  3.03293824e-01  -1.31169760e+00  -1.89557159e+00   1.93452710e-04]
 [ -4.75692600e-01   1.09454644e+00   1.35424197e+00  -3.89351398e-01]
 [ -9.90841687e-01   6.82334840e-01   1.15010154e+00   7.63322830e-01]]
[[ 0.32943594  0.07537079  0.89375246  0.22527957]
 [ 0.94404054  0.32063842  0.90321803  0.42380559]
 [ 0.53356743  0.12545431  0.67474759  0.25374627]
 [ 0.70735872  0.46039164  0.10329127  0.94251609]]
```

Many sessions have op.eval\(\) and you can get the same results. 

**Create Vector**

```py
b = tf.constant([ [10], [100]])
b.get_shape()

TensorShape([Dimension(2), Dimension(1)])
```

**Create a 2x2 Matrix **

```py
a = tf.constant([ [1,2],
                  [3,4]  ])
a.get_shape()

TensorShape([Dimension(2), Dimension(2)])
```

**Multiply a Vector and Matrix**

```py
result = tf.matmul(a,b)
sess.run(result)

array([[210],
       [430]], dtype=int32)
```




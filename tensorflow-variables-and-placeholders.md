

# TensorFlow

They're two types of Tensor Objects

* **Variables**
  * Can hold the weight and biases throughout a session.
  * They need to be initialized 
* **Placeholders**
  * Initially empty
  * Used to feed in the actual training examples
  * Requirements:
    * Data Type \(tf.float32\)
    * Shape of Data

**Create Random Tensor**

```py
import tensorflow as tf
sess = tf.InteractiveSession()
my_tensor = tf.random_uniform((4,4), 0, 1)
```

tf.random\_uniform\(size-of-matrix, minValue, maxValue\)

```py
print(my_tensor)
```

Output

```
<tf.Tensor 'random_uniform:0' shape=(4, 4) dtype=float32>
```

_Notice this is of dtype=float32._

---

Now let's **Set my\_tensor to TensorFlow type variable.**

```py
my_var = tf.Variable(initial_value=my_tensor)
```

Run it

```py
sess.run(my_var)
```

Output

```py
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/anaconda3/envs/tfdeeplearning/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1326     try:
-> 1327       return fn(*args)
   1328     except errors.OpError as e:
---snip---

---snip---
/anaconda3/envs/tfdeeplearning/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1338         except KeyError:
   1339           pass
-> 1340       raise type(e)(node_def, op, message)
   1341 
   1342   def _extend_graph(self):

FailedPreconditionError: Attempting to use uninitialized value Variable
	 [[Node: _retval_Variable_0_0 = _Retval[T=DT_FLOAT, index=0, _device="/job:localhost/replica:0/task:0/cpu:0"](Variable)]]

```

Looks like we got a FailedPreconditionError....error. We need to initialize our variables first.

**Initialize Variables**

```
init = tf.global_variables_initializer()
sess.run(init)
```




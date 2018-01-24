# TensorFlow Graphs

Graphs are sets of connected **nodes**/**vertices**.  
Two nodes that are connected, the connection itself is called an **edge**.  
Let's create the following graph.

![](/assets/Screen Shot 2018-01-23 at 5.10.40 PM.png)

```py
import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
print(result)
```

Output

```
3
```

Not bad at all!

---

When you start TF, a default graph is created.![](/assets/Screen Shot 2018-01-23 at 5.14.12 PM.png)

```
print(tf.get_default_graph())
```

Output

```
<tensorflow.python.framework.ops.Graph object at 0x116702550>
```

---

**Create Graph Object \(empty\)**

```
g = tf.Graph()
print(g)
```

Output

```
<tensorflow.python.framework.ops.Graph object at 0x1167b4278>
```

**Set Default Graph to a Variable**  
In this first case, we will set the default graph to graph\_one.

```py
graph_one = tf.get_default_graph()
print(graph_one)
```

Output

```
<tensorflow.python.framework.ops.Graph object at 0x116702550>
```

**Change Graph to your Default Graph**

```py
with g.as_default():
    print(g is tf.get_default_graph()
```

Output

```py
True
```

This comes into play when we want to reset our graph in our notebook. It's not usually used.


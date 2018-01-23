# NumPy Crash Course

You can create arrays. Yay!

```
import numpy as np
my_list = [1,2,3]
np.array(my_list)
```

**Output**

```
array([1, 2, 3])
```

_This is of type **numpy.ndarry** which stands for n-dimensional array._



**Create a range of ints in an array.**

```
np.arange(0,11,2)
```

Output

```
array([ 0,  2,  4,  6,  8, 10])
```



**Create a matrix of zeros.**

```
np.zeros((3,5))
```

Output

```
array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
```

_They're floats, which is why there's a "."_



**Return an evenly spaced set of numbers over a specified interval.**

```
np.linspace(0,9,3)
```

Output

```
array([ 0. ,  4.5,  9. ])
```



**Create an array of random integers**

```
np.random.randint(0,1000,(3,3))
```

Output

```
array([[321, 831, 601],
       [729, 951, 597],
       [389, 923, 644]])
```




# NumPy Crash Course

**Create an Array**

```py
import numpy as np
my_list = [1,2,3]
np.array(my_list)

array([1, 2, 3])
```

_This is of type **numpy.ndarry** which stands for n-dimensional array._

**Create a range of ints in an array.**

```py
np.arange(0,11,2)

array([ 0,  2,  4,  6,  8, 10])
```

**Create a matrix of zeros.**

```py
np.zeros((3,5))

array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
```

_They're floats, which is why there's a "."_

**Return an evenly spaced set of numbers over a specified interval.**

```py
np.linspace(0,9,3)

array([ 0. ,  4.5,  9. ])
```

**Create an array of random integers**

```py
np.random.randint(0,1000,(3,3))

array([[321, 831, 601],
       [729, 951, 597],
       [389, 923, 644]])
```

**Set random seed and generate random numbers**

```py
np.random.seed(101)
np.random.randint(0,100,10)

array([95, 11, 81, 70, 63, 87, 75,  9, 77, 40])
```

**Find the Max Values of an Array**

```py
arr = np.random.randint(0,100,10)
arr.max()

93
```

**Find the Min Values of an Array**

```py
arr.min()

4
```

**Find the element number on the max integer in an array**

```py
arr.argmax()

8
```

**Reshape an array into a a 2x5 Matrix**

```py
arr.reshape(2,5)

array([[ 4, 63, 40, 60, 92],
       [64,  5, 12, 93, 40]])
```

**Create a Matrix from range\(0,99\)**

```py
mat = np.arange(0,100).reshape(10,10)
mat

array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
```

**Print Element in an Array**

```py
mat[0,1]

1
```

**Print a Row**

```py
mat[:,0]


array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
```

**Print a Column**

```py
mat[0,:]

array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**Print/Slice a section of the Matrix**

```py
mat[0:3,0:3]

array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22]])
```




# SciKit-Learn Crash Course

Using this for pre-processing

1. Scaling data
2. Split it into different test sets

**Imports**

```py
from sklearn.preprocessing import MinMaxScaler
```

**Create Data**

```py
data = np.random.randint(0,100,(10,2))
data

array([[45, 55],
       [11, 40],
       [ 4, 15],
       [20, 21],
       [71,  4],
       [29, 78],
       [74, 27],
       [85, 56],
       [55,  2],
       [22, 18]])
```

We can think of this as features/labels in a dataset.  
If we want to run this into a neural networks, we need to scale the data. We will do this by creating an instsance of a MinMax Scaler.

**Create a MinMaxScaler Object**

```py
scaler_model = MinMaxScaler()
```

Now let's fit and transform the our data.  
This **normalizes** our data from range \(0,1\). This can also be adjusted.

```py
scaler_model.fit(data)
scaler_model.transform(data)
```

```py
array([[ 0.50617284,  0.69736842],
       [ 0.08641975,  0.5       ],
       [ 0.        ,  0.17105263],
       [ 0.19753086,  0.25      ],
       [ 0.82716049,  0.02631579],
       [ 0.30864198,  1.        ],
       [ 0.86419753,  0.32894737],
       [ 1.        ,  0.71052632],
       [ 0.62962963,  0.        ],
       [ 0.22222222,  0.21052632]])
```

One can also do this in one command

```py
scaler_model.fit_transform(data)
```

```py
array([[ 0.50617284,  0.69736842],
       [ 0.08641975,  0.5       ],
       [ 0.        ,  0.17105263],
       [ 0.19753086,  0.25      ],
       [ 0.82716049,  0.02631579],
       [ 0.30864198,  1.        ],
       [ 0.86419753,  0.32894737],
       [ 1.        ,  0.71052632],
       [ 0.62962963,  0.        ],
       [ 0.22222222,  0.21052632]])
```

**Create a df DataFrame**

```py
import pandas as pd
mydata = np.random.randint(0,101,(50,4))
data = pd.DataFrame(data=mydata)
data

    0    1    2    3
0    95    79    45    76
1    0    50    85    20
2    15    5    23    65
3    38    92    37    39
4    16    42    56    74
5    46    4    61    21
6    48    31    31    70
7    14    34    47    33
8    89    74    62    78
9    76    46    63    54
---snip---
```

**Label DataFrame**

```py
df = pd.DataFrame(data=mydata,columns = ['f1', 'f2', 'f3', 'label'])
df

    f1    f2    f3    label
0    93    94    79    40
1    12    96    62    28
2    90    97    67    62
3    88    65    27    56
4    3    92    9    46
5    29    82    98    74
6    52    40    72    91
7    67    19    55    51
8    3    78    92    50
9    87    53    60    45
10    91    60    44    90
---snip---
```

**Create a Matrix of Features \(X\)**

```py
X = df[['f1', 'f2', 'f3']]
X

    f1    f2    f3
0    93    94    79
1    12    96    62
2    90    97    67
3    88    65    27
4    3    92    9
5    29    82    98
6    52    40    72
7    67    19    55
8    3    78    92
9    87    53    60
10    91    60    44
--snip--
```

**Create a Matrix of Labels \(y\)**

```py
y = df['label']
y

0      40
1      28
2      62
3      56
4      46
5      74
6      91
7      51
8      50
9      45
10     90
--snip--
```

**Split Training Matrix**

To get an example of train\_test\_split is to type it out, then press SHIFT+TAB, TAB.

\_Here is the train set.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.shape

(35, 3)
```

Here is the test.

```
X_test.shape

(15, 3)
```

Once we have the neural network model working in TensorFlow, and we want to do some training process on supervised learning, we would feed it in the training sets for X-train and y-train, and the model would then try to predict how it got the results of y-train.The model would try to build some sort of understanding. Once we have that, we can evlaute that by feeding it the X-Test data, and try to predict. We can then check to see how close they are to the actual values of y. That's the reason for the train\_test\_split\(\).


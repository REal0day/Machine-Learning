# SciKit-Learn

Using this for pre-processing

1. Scaling data
2. Split it into different test sets

**Imports**

```
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
type(scaler_model)

sklearn.preprocessing.data.MinMaxScaler
```

Now let's fit and transform the our data.  
This normalizes our data from range \(0,1\). This can also be adjusted.

```py
scaler_model.fit(data)
scaler_model.transform(data)

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




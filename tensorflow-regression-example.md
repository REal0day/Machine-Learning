# TensorFlow Regression Example

We will now use a more realistic regression example and introduce **tf.estimator**.  
Let's start with our imports.

```py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
```

Sweet. Let's create a "HUGE" dataset_. Huge is relative. This will have 1M datapoints._

```py
x_data = np.linspace(0.0,10.0,1000000)
```

Now let's create the same amount of noise as our dataset. \(1M\)

```py
noise = np.random.randn(len(x_data))
```

Our line will be modeled as y = mx + b. We will set b = 5 and add noise.

```py
y_true = (0.5 * x_data) + 5 + noise
```

_The noise is added so it's our true line is not just a perfect fitted line. The 5 represents the shift in b that we want._

Let's use pandas to concatenate our data.

**Create our x\_dataFrame**

```py
x_df = pd.DataFrame(data=x_data, columns=['X Data'])
```

**Create our y\_dataFrame**

```py
y_df = pd.DataFrame(data=y_true, columns=['Y'])
```

Now let's check the first five entries of x\_df

```py
x_df.head()
```

Output

```py
     X Data
0    0.00000
1    0.00001
2    0.00002
3    0.00003
4    0.00004
```

Now let's **concat** them both to have one frame!

```
my_data = pd.concat([x_df,y_df], axis=1)
```

Output of my\_data.head\(\)

```py
     X Data     Y
0    0.00000    5.718261
1    0.00001    5.000671
2    0.00002    5.544956
3    0.00003    5.070396
4    0.00004    4.691148
```

_We add the axis=1 so that the data isn't stacked like a pancake!_

Now let's say we want to graph this. Unfortunately, graphing a million plot points will take awhile.  
Let's only graph a small random sample of 10.

```py
my_data.sample(n=10)
```

Output

```py
index	X Data		Y
140413	1.404131	5.637388
763166	7.631668	9.500907
210459	2.104592	5.871338
238403	2.384032	6.298123
74718	0.747181	5.466532
68900	0.689001	4.138463
691081	6.910817	9.494513
797712	7.977128	7.670427
517287	5.172875	6.555583
63754	0.637541	6.250593 
```




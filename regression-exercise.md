# TensorFlow Regression Exercise

It will take place in the following steps:

1. **Collect Data & Create Variables**
   1. **X\_data** = All values to be used for calculations
   2. **y\_labels** = All answers of those values after calculations
2. **Splice X\_data into X\_train and X\_test**
   1. **X\_train** will be used as values to train your model. After, it will be evaluated with **X\_test**, and will receive a percentage of accuracy in terms of how well it performed/how close the values from **X\_test** were to their corresponding **y\_labels**
3. **Scale the Feature Data**  
   1. Use sklearn preprocessing to create a MinMaxScaler for the feature data. **Fit this scaler only to the training data**. Then use it to transform X\_test and X\_train. Then use the scaled X\_test and X\_train along with pd.Dataframe to re-create two dataframes of scaled data.

   1. **Remember!!: DO NOT USE THE SCALER ON THE X\_train data. **When you test your model, you don't want it believing it'll have more data...such as X\_train.

## The Data

#### Imports

```py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
%matplotlib inline
```

#### Open CSV as a pandas DataFrame

```py
houses = pd.read_csv('cal_housing_clean.csv')
```

#### Create X\_data without y\_labels

```py
X_data = houses.drop('medianHouseValue', axis=1)
```

#### Create y\_labels

```py
y_labels = houses['medianHouseValue']
```

#### Split X\_Data and create X\_train\(70%\) and X\_test\(30%\)

```py
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.3,random_state=101)
```

_Applied random\_state but that's option. _

#### Describe your Data

```py
X_data.describe()
```

```py
    housingMedianAge    totalRooms    totalBedrooms    population    households    medianIncome
count    20640.000000    20640.000000    20640.000000    20640.000000    20640.000000    20640.000000
mean    28.639486    2635.763081    537.898014    1425.476744    499.539680    3.870671
std    12.585558    2181.615252    421.247906    1132.462122    382.329753    1.899822
min    1.000000    2.000000    1.000000    3.000000    1.000000    0.499900
25%    18.000000    1447.750000    295.000000    787.000000    280.000000    2.563400
---snip---
```

## Scale the Feature Data

#### Create Scaler Model to normailze the data \[0,1\]

```py
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
```

#### Fit Scaler Model with X\_train Data

**Remember!!: DO NOT USE THE SCALER ON THE X\_train data. **When you test your model, you don't want it believing it'll have more data...such as X\_train.

```py
trained_scaler = scaler_model.fit(X_train)
```

#### Create pd DataFrame for both Scaler X\_train and Scaler X\_test

This isn't making the scaler of these matrices into our Model. Only X\_train has been fitted into our model.

```py
scaled_X_train = scaler_model.transform(X_train)
pdscaled_X_train = pd.DataFrame(data=scaled_X_train)

scaled_X_test = scaler_model.transform(X_test)
pdscaled_X_test = pd.DataFrame(data=scaled_X_test)
```

## Create Feature Columns






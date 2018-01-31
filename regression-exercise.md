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
housing = pd.read_csv('cal_housing_clean.csv')
```

#### Create X\_data without y\_labels

```py
x_data = housing.drop(['medianHouseValue'],axis=1)
```

#### Create y\_labels

```py
y_val = housing['medianHouseValue']
```

#### Split X\_Data and create X\_train\(70%\) and X\_test\(30%\)

```py
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)
```

_Applied random\_state but that's option. _

#### Describe your Data

```py
housing.describe().transpose()
```

```py
count    mean    std    min    25%    50%    75%    max
housingMedianAge    20640.0    28.639486    12.585558    1.0000    18.0000    29.0000    37.00000    52.0000
totalRooms    20640.0    2635.763081    2181.615252    2.0000    1447.7500    2127.0000    3148.00000    39320.0000
totalBedrooms    20640.0    537.898014    421.247906    1.0000    295.0000    435.0000    647.00000    6445.0000
population    20640.0    1425.476744    1132.462122    3.0000    787.0000    1166.0000    1725.00000    35682.0000
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
scaler = MinMaxScaler()
scaler.fit(X_train)
```

#### Create pd DataFrame for both Scaler X\_train and Scaler X\_test

This isn't making the scaler of these matrices into our Model. Only X\_train has been fitted into our model.

```py
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)
```

## Create Feature Columns

**View all Columns**

```py
housing.columns
```

Create the necessary** tf.feature\_column** objects for the **estimator**. They should all be trated as **continuous numeric\_columns**.

```py
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [ age,rooms,bedrooms,pop,households,income]
```

**Create the input function** for the **estimator object**. \(play around with batch\_size and num\_epochs\)

```py
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
                                            shuffle=True)
```

**Create the Estimator Model.** Use a **DNNRegressor**. Play around with the hidden units!

```py
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)
```

#### **Train Model**

```py
model.train(input_fn=input_func,steps=25000)
```

#### Get values for prediction

```py
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)
```

#### Calculate the RMSE

You can do this manually or use sklearn.metrics like so.

```
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])
```

```
from sklearn.metrics import mean_squared_error
```

```
mean_squared_error(y_test,final_preds)**0.5
```




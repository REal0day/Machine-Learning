# Pandas Crash Course

Pandas is written from NumPy.

**Get data from CSV**

```py
df = pd.read_csv('salaries.csv')
df
```

|  |
| :--- |


|  | Name | Salary | Age |
| :--- | :--- | :--- | :--- |
| 0 | John | 50000 | 34 |
| 1 | Sally | 120000 | 45 |
| 2 | Alyssa | 80000 | 27 |

**Describe Data**

```py
df.describe()
```

|  |
| :--- |


| Salary | Age |
| :--- | :--- |


| count | 3.000000 | 3.000000 |
| :--- | :--- | :--- |


| mean | 83333.333333 | 35.333333 |
| :--- | :--- | :--- |


| std | 35118.845843 | 9.073772 |
| :--- | :--- | :--- |


| min | 50000.000000 | 27.000000 |
| :--- | :--- | :--- |


| 25% | 65000.000000 | 30.500000 |
| :--- | :--- | :--- |


| 50% | 80000.000000 | 34.000000 |
| :--- | :--- | :--- |


| 75% | 100000.000000 | 39.500000 |
| :--- | :--- | :--- |


| max | 120000.000000 | 45.000000 |
| :--- | :--- | :--- |


**Print all Rows where Column "Salary" &gt; 60000**

```py
df[df['Salary'] > 60000]
```

|  |
| :--- |


|  | Name | Salary | Age |
| :--- | :--- | :--- | :--- |
| 1 | Sally | 120000 | 45 |
| 2 | Alyssa | 80000 | 27 |

**Return NumPy Array**

```py
df.as_matrix()

array([['John', 50000, 34],
       ['Sally', 120000, 45],
       ['Alyssa', 80000, 27]], dtype=object)
```

**Create a DataFrame using your data**

```py
mat = np.random.randint(1,101,(100,5))
df = pd.DataFrame(data=mat)

0	1	2	3	4
0	96	12	82	71	64
1	88	76	10	78	41
2	5	64	41	61	93
3	65	6	13	94	41
4	50	84	9	30	60
5	35	45	73	20	11
6	77	96	88	1	74
7	9	63	37	84	100
8	29	64	8	11	53
9	57	39	74	53	19
10	72	16	45	1	13
```




##### 窗口函数

Window Functions 有下列的属性

- 在一组行上面执行计算，这一组行称为Frame
- 每行row对应一个Frame
- 给每行返回一个新的值通过aggregate/window 函数

在spark函数中，只有Aggregate Functions 能够和 Window Functions搭配使用

Basic Frame有下列的属性

- 被一列或者多列的Window.partitionBy生成
- 每行对应一个Frame
- Frame在同一个partition里面是相同的
- Aggregate/Window functions 可以运用于每个row+frame 去生成单个的值

```python
overCategory = Window.partitionBy("depName")
df = empsalary.withColumn(
"salaries", collect_list("salary").over(overCategory)).withColumn(
"avg",(avg("salary").over(overCategory)).cast("int")).withColumn(
"tot",sum("salary").over(overCategory))
```

| depName   | name   | salary | salaries                   | avg  | total |
| --------- | ------ | ------ | -------------------------- | ---- | ----- |
| develop   | Saga   | 6000   | [6000]                     | 6000 | 6000  |
| develop   | Wilma  | 5200   | [6000,5200,5200]           | 5466 | 16400 |
| develop   | Maja   | 5200   | [6000,5200,5200]           | 5466 | 16400 |
| develop   | Freja  | 4500   | [6000,5200,5200,4500]      | 5225 | 20900 |
| develop   | Astrid | 4200   | [6000,5200,5200,4500,4200] | 5020 | 25100 |
| sales     | Alice  | 5000   | [5000]                     | 5000 | 5000  |
| sales     | Ella   | 4800   | [5000,4800,4800]           | 4866 | 14600 |
| sales     | Ebba   | 4800   | [5000,4800,4800]           | 4866 | 14600 |
| personnel | Olivia | 3900   | [3900]                     | 3900 | 3900  |
| personnel | Lilly  | 3500   | [3900,3500]                | 3700 | 7400  |

Ordered Frame has the following traits

- 被一个或者是多个columns生成
- Followed by orderby on a column
- Each row have a corresponding frame
- The frame will not be the same for every row within the same `partition.By` default,the frame contains all previous rows and the currentRow
- Aggregate/Window functions can be applied to each row+frame to generate a value

```python
overCategory = Window.partitionBy("depName").orderBy(desc("salary"))
df = empsalary.withColumn(
"salaries",collect_list("salary").over(overCategory)).withColumn(
"average_salary",(avg("salary").over(overCategory)).cast("int")).withColumn(
"total_salary",sum("salary").over(overCategory))
```

| Function       | Description             | Note                                                         |
| -------------- | ----------------------- | ------------------------------------------------------------ |
| `rank`         | rank of rows            | `20,30,30,40`经过rank后的序号为`1,2,2,4`                     |
| `dense_rank`   | dense rank of rows      | `20,30,30,40`经过rank后的序号为`1,2,2,3`                     |
| `row_number`   | row number              | `20,30,30,40`经过rank后的序号为`1,2,3,4`                     |
| `ntile`        | ntile id                | split partition to ntile the first group in the window be tile 1. |
| `percent_rank` | (rank-1)/(total_rows-1) | useful like take 20% top people                              |

```python
overCategory = window.partitionBy("depName").orderBy(desc("salary"))
df = empsalary.withColumn(
"salaries",collect_list("salary").over(overCategory)).withColumn(
"rank",rank().over(overCategory)).withColumn(
"dense_rank",dense_rank().over(overCategory)).withColumn(
"row_number",row_number().over(overCategory)).withColumn(
"ntile",ntile(3).over(overCategory)).withColumn(
"percent_rank",percent_rank().over(overCategory))
```

| depName   | name   | salary | rank | dense_rank | row_number | ntile | percent_rank |
| --------- | ------ | ------ | ---- | ---------- | ---------- | ----- | ------------ |
| develop   | Saga   | 6000   | 1    | 1          | 1          | 1     | 0.0          |
| develop   | Wilma  | 5200   | 2    | 2          | 2          | 1     | 0.25         |
| develop   | Maja   | 5200   | 2    | 2          | 3          | 2     | 0.25         |
| develop   | Freja  | 4500   | 4    | 3          | 4          | 2     | 0.75         |
| develop   | Astrid | 4200   | 5    | 4          | 5          | 3     | 1.0          |
| sales     | Alice  | 5000   | 1    | 1          | 1          | 1     | 0.0          |
| sales     | Ella   | 4800   | 2    | 2          | 2          | 2     | 0.5          |
| sales     | Ebba   | 4800   | 2    | 2          | 3          | 3     | 0.5          |
| personnel | Olivia | 3900   | 1    | 1          | 1          | 1     | 0.0          |
| personnel | Lilly  | 3500   | 2    | 2          | 2          | 2     | 1.0          |

lag 的意思是拿到当前value的前一个。lead 的意思是拿到当前value的后一个

```python
df = empsalary.withColumn(
"lead",lead("salary",1).over(overCategory)).withColumn(
"lag",lag("salary",1).over(overCategory))
```

| depName   | name   | salary | lead | lag  |
| --------- | ------ | ------ | ---- | ---- |
| develop   | Saga   | 6000   | 5200 | null |
| develop   | Wilma  | 5200   | 5200 | 6000 |
| develop   | Maja   | 5200   | 4500 | 5200 |
| develop   | Freja  | 4500   | 4200 | 5200 |
| develop   | Astrid | 4200   | null | 4500 |
| sales     | Alice  | 5000   | 4800 | null |
| sales     | Ella   | 4800   | 4800 | 5000 |
| sales     | Ebba   | 4800   | null | 4800 |
| personnel | Olivia | 3900   | 3500 | null |
| personnel | Lilly  | 3500   | null | 3500 |

###### Range Frame

利用range functions去改变frame的边界（boundary）

- 被生成通过Window.partitionBy 一列或者多列
- 通常是有orderBy的，所以在frame里面的数据是被排序过的
- Then followed by rangeBetween or rowsBetween
- 每行对应一个frame
- frame的边界是被rangeBetween 和 rowsBetween控制的
- Aggregate/Window functions可以被应用到row+frame上去生成单个的值

rowsBetween get the frame boundary based on the row index in the window compared to currentRow

| example                                                      | meaning                                        |
| ------------------------------------------------------------ | ---------------------------------------------- |
| `.rowsBetween(Window.currentRow, 1)`                         | currentRow and the next row                    |
| `.rowsBetween(Window.currentRow, 2)`                         | currentRow and the next 2 rows                 |
| `.rowsBetween(-1, Window.currentRow)`                        | previous row and the currentRow                |
| `.rowsBetween(-1, 1)`                                        | previous row, the current row and the next row |
| `.rowsBetween(Window.unboundedPreceding, Window.currentRow)` | all previous rows and the currentRow           |
| `.rowsBetween(Window.currentRow,Window.unboundedFollowing)`  | all next rows and the currentRow               |
| `.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)` | all rows in the window                         |

rangeBetween 拿到frame的边界基于window内的row value，the difference compares to rowsBetween is that it compare with value of the current row

```python
Window.currentRow = 0
Window.unboundedPreceding = Long.MinValue
Window.unboundedFollowing = Long.MaxValue
```


A PySpark DataFrame can be created via `pyspark.sql.SparkSession.createDataFrame` typically by 
passing a list of lists, tuples, dictionaries and `pyspark.sql.Rows`, a pandas DataFrame and an 
RDD consisting of such a list. `pyspark.sql.SparkSession.createDataFrame` takes the `schema` argument 
to specify the schema of the DataFrame. When it is omitted, PySpark infers the corresponding schema by taking a sample from the data.
```python
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

df = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')

rdd = spark.sparkContext.parallelize([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
])
df = spark.createDataFrame(rdd, schema=['a', 'b', 'c', 'd', 'e'])
```
PySpark DataFrame is lazily evaluated and simply selecting a column does not trigger the computation but it returns a `Column` instance.
most of column-wise operations return `Columns`.
These Columns can be used to select the columns from a DataFrame. For example, DataFrame.select() takes the Column instances that returns another DataFrame.
Assign new Column instance.
##### Applying a Function
```python
import pandas
from pyspark.sql.functions import pandas_udf

@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply plus one by using pandas Series.
    return series + 1

df.select(pandas_plus_one(df.a)).show()
```
Another example is DataFrame.mapInPandas which allows users directly use the APIs in a pandas DataFrame without any restrictions such as the result length.
```python
def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.a == 1]

df.mapInPandas(pandas_filter_func, schema=df.schema).show()
```
##### Grouping Data
```python
df = spark.createDataFrame([
    ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
df.show()
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana|  1| 10|
| blue|banana|  2| 20|
|  red|carrot|  3| 30|
| blue| grape|  4| 40|
|  red|carrot|  5| 50|
|black|carrot|  6| 60|
|  red|banana|  7| 70|
|  red| grape|  8| 80|
+-----+------+---+---+

def plus_mean(pandas_df):
    return pandas_df.assign(v1=pandas_df.v1 - pandas_df.v1.mean())

df.groupby('color').applyInPandas(plus_mean, schema=df.schema).show()
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana| -3| 10|
|  red|carrot| -1| 30|
|  red|carrot|  0| 50|
|  red|banana|  2| 70|
|  red| grape|  3| 80|
|black|carrot|  0| 60|
| blue|banana| -1| 20|
| blue| grape|  1| 40|
+-----+------+---+---+

#Co-grouping and applying a function.

df1 = spark.createDataFrame(
    [(20000101, 1, 1.0), (20000101, 2, 2.0), (20000102, 1, 3.0), (20000102, 2, 4.0)],
    ('time', 'id', 'v1'))

df2 = spark.createDataFrame(
    [(20000101, 1, 'x'), (20000101, 2, 'y')],
    ('time', 'id', 'v2'))

def asof_join(l, r):
    return pd.merge_asof(l, r, on='time', by='id')

df1.groupby('id').cogroup(df2.groupby('id')).applyInPandas(
    asof_join, schema='time int, id int, v1 double, v2 string').show()
+--------+---+---+---+
|    time| id| v1| v2|
+--------+---+---+---+
|20000101|  1|1.0|  x|
|20000102|  1|3.0|  x|
|20000101|  2|2.0|  y|
|20000102|  2|4.0|  y|
+--------+---+---+---+

```
#####Working with SQL
```python
df.createOrReplaceTempView("tableA")
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.udf.register("add_one", add_one)
spark.sql("SELECT add_one(v1) FROM tableA").show()
```
##### Arrow in spark
Arrow is available as an optimization when converting a Spark DataFrame to a Pandas DataFrame using the call `DataFrame.toPandas()` and when creating a Spark DataFrame from a Pandas DataFrame with `SparkSession.createDataFrame()`.
To use Arrow when executing these calls, users need to first set the Spark configuration `spark.sql.execution.arrow.pyspark.enabled` to `true`. This is disabled by default.
######Pandas UDFs
Pandas UDFs are user defined functions that are executed by Spark using Arrow to transfer data and Pandas to work with the data, which allows vectorized operations. A Pandas UDF is defined using the pandas_udf() as a decorator or to wrap the function, and no additional configuration is required. A Pandas UDF behaves as a regular PySpark function API in general.

Before Spark 3.0, Pandas UDFs used to be defined with pyspark.sql.functions.PandasUDFType. From Spark 3.0 with Python 3.6+, you can also use Python type hints. Using Python type hints is preferred and using pyspark.sql.functions.PandasUDFType will be deprecated in the future release.

Note that the type hint should use pandas.Series in all cases but there is one variant that pandas.DataFrame should be used for its input or output type hint instead when the input or output column is of StructType. The following example shows a Pandas UDF which takes long column, string column and struct column, and outputs a struct column. It requires the function to specify the type hints of pandas.Series and pandas.DataFrame as below:
```python
import pandas as pd

from pyspark.sql.functions import pandas_udf

@pandas_udf("col1 string, col2 long")
def func(s1: pd.Series, s2: pd.Series, s3: pd.DataFrame) -> pd.DataFrame:
    s3['col2'] = s1 + s2.str.len()
    return s3

# Create a Spark DataFrame that has three columns including a struct column.
df = spark.createDataFrame(
    [[1, "a string", ("a nested string",)]],
    "long_col long, string_col string, struct_col struct<col1:string>")

df.printSchema()
# root
# |-- long_column: long (nullable = true)
# |-- string_column: string (nullable = true)
# |-- struct_column: struct (nullable = true)
# |    |-- col1: string (nullable = true)

df.select(func("long_col", "string_col", "struct_col")).printSchema()
# |-- func(long_col, string_col, struct_col): struct (nullable = true)
# |    |-- col1: string (nullable = true)
# |    |-- col2: long (nullable = true)
```
```python

from pyspark.sql.functions import pandas_udf, PandasUDFType
# Use pandas_udf to define a Pandas UDF
@pandas_udf('double', PandasUDFType.SCALAR)
# Input/output are both a pandas.Series of doubles
def pandas_plus_one(v):
    return v+1
df.withColumn('v2',pandas_plus_one(df.v))

@pandas_udf(df.schema,PandasUDFType.GROUPED_MAP)
##Input/output are both a pandas.DataFrame
def subtract_mean(pdf):
    return pdf.assign(v = pdf.v - pdf.v.mean()

df.groupby('id').apply(subtract_mean)

import statsmodels.api as sm
## df has four columns : id ,y ,x1, x2

group_column = 'id'
y_column = 'y'
x_columns = ['x1','x2']
schema = df.select(group_column,*x_columns).schema

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y,X).fit()
    return pd.DataFrame([[group_key] + [model.params[i] for i in   x_columns]], columns=[group_column] + x_columns)
beta = df.groupby(group_column).apply(ols)
```
可能存在的问题：用pyspark，pandas udf，老是被yarn kill掉container，说是物理内存超了

原因：很有可能是spark里的partition过大，转换到pandas dataframe的时候导致内存爆炸。假设你spark的一个dataframe，虽然只有6g，但是只有6个partition，每个partition有1g，那么pandas_udf会把一整个partition转成pandas dataframe然后调用python来处理。好家伙，pandas的dataframe内存使用效率可低了。spark里1g的partition变成pandas dataframe可能有10g。6个partition并行，就是60g内存，直接爆炸了。

#####UDF
```python
from pyspark.sql.types import IntegerType
slen = udf(lambda s: len(s), IntegerType())
@udf
def to_upper(s):
    if s is not None:
        return s.upper()

@udf(returnType=IntegerType())
def add_one(x):
    if x is not None:
        return x + 1

df = spark.createDataFrame([(1, "John Doe", 21)], ("id", "name", "age"))
df.select(slen("name").alias("slen(name)"), to_upper("name"), add_one("age")).show()

```
####RDD Programming
Spark supports two types of shared variables: broadcast variables, accumulators, which are variables that are only “added” to, such as counters and sums.
```scala
val conf = new SparkConf().setAppName(appName).setMaster(master)
val sc = new SparkContext(conf)
```

There are two ways to create RDDs: parallelizing an existing collection in your driver program, or referencing a dataset in an external storage syste

```scala
val data = Array(1, 2, 3, 4, 5)
val distData = sc.parallelize(data)
```


Some notes on reading files with Spark:

- If using a path on the local filesystem, **the file must also be accessible at the same path on worker nodes**. Either copy the file to all workers or use a network-mounted shared file system.

##### Operation
RDDs support two types of operations: transformations, which create a new dataset from an existing one, and actions, which return a value to the driver program after running a computation on the dataset. 

By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also persist an RDD in memory using the `persist` (or `cache`) method, in which case Spark will keep the elements around on the cluster for much faster access the next time you query it. There is also support for persisting RDDs on disk, or replicated across multiple nodes.

```scala
val lines = sc.textFile("data.txt")
val lineLengths = lines.map(s => s.length)
val totalLength = lineLengths.reduce((a, b) => a + b)
```

If we also wanted to use lineLengths again later, we could add:
```scala
lineLengths.persist()
```
###### Passing Functions to Spark
Spark’s API relies heavily on passing functions in the driver program to run on the cluster. There are two recommended ways to do this:

- Anonymous function syntax, which can be used for short pieces of code.
- Static methods in a global singleton object. For example, you can define object MyFunctions and then pass MyFunctions.func1

```scala
object MyFunctions {
  def func1(s: String): String = { ... }
}

myRdd.map(MyFunctions.func1)
```
```python
"""MyScript.py"""
if __name__ == "__main__":
    def myFunc(s):
        words = s.split(" ")
        return len(words)

    sc = SparkContext(...)
    sc.textFile("file.txt").map(myFunc)
```
Here, if we create a new MyClass instance and call doStuff on it, the map inside there references the func1 method of that MyClass instance, so the whole object needs to be sent to the cluster. It is similar to writing rdd.map(x => this.func1(x)).

In a similar way, accessing fields of the outer object will reference the whole object:
```scala
class MyClass {
  val field = "Hello"
  def doStuff(rdd: RDD[String]): RDD[String] = { rdd.map(x => field + x) }
}
```
```python
class MyClass(object):
    def func(self, s):
        return s
    def doStuff(self, rdd):
        return rdd.map(self.func)
```
is equivalent to writing `rdd.map(x => this.field + x)`, which references all of this. To avoid this issue, the simplest way is to copy field into a local variable instead of accessing it externally:
```scala
def doStuff(rdd: RDD[String]): RDD[String] = {
  val field_ = this.field
  rdd.map(x => field_ + x)
}
```
##### Closures
One of the harder things about Spark is understanding the scope and life cycle of variables and methods when executing code across a cluster. RDD operations that modify variables outside of their scope can be a frequent source of confusion

```scala
var counter = 0
var rdd = sc.parallelize(data)

// Wrong: Don't do this!!
rdd.foreach(x => counter += x)

println("Counter value: " + counter)
```
The behavior of the above code is undefined, and may not work as intended. To execute jobs, Spark breaks up the processing of RDD operations into tasks, each of which is executed by an executor. Prior to execution, Spark computes the task’s closure. The closure is those variables and methods which must be visible for the executor to perform its computations on the RDD (in this case foreach()). This closure is serialized and sent to each executor.

The variables within the closure sent to each executor are now copies and thus, 
when counter is referenced within the foreach function, it’s no longer the counter on the driver node. 
**There is still a counter in the memory of the driver node but this is no longer visible to the executors!** The executors only see the copy from the serialized closure. Thus, the final value of counter will still be zero since all operations on counter were referencing the value within the serialized closure.

In local mode, in some circumstances, the foreach function will actually execute within the same JVM as the driver and will reference the same original counter, and may actually update it.

To ensure well-defined behavior in these sorts of scenarios one should use an Accumulator. Accumulators in Spark are used specifically to provide a mechanism for safely updating a variable when execution is split up across worker nodes in a cluster. 

In general, closures - constructs like loops or locally defined methods, should not be used to mutate some global state. Spark does not define or guarantee the behavior of mutations to objects referenced from outside of closures. Some code that does this may work in local mode, but that’s just by accident and such code will not behave as expected in distributed mode. Use an Accumulator instead if some global aggregation is needed.
######Printing elements of an RDD
Another common idiom is attempting to print out the elements of an RDD using rdd.foreach(println) or rdd.map(println). On a single machine, this will generate the expected output and print all the RDD’s elements. However, in cluster mode, the output to stdout being called by the executors is now writing to the executor’s stdout instead, not the one on the driver, so stdout on the driver won’t show these! To print all elements on the driver, one can use the collect() method to first bring the RDD to the driver node thus: rdd.collect().foreach(println). 
#####Key-Value Pairs
The key-value pair operations are available in the PairRDDFunctions class, which automatically wraps around an RDD of tuples.

In Scala, these operations are automatically available on RDDs containing Tuple2 objects (the built-in tuples in the language, created by simply writing (a, b)).

For example, the following code uses the reduceByKey operation on key-value pairs to count how many times each line of text occurs in a file:
```scala
val lines = sc.textFile("data.txt")
val pairs = lines.map(s => (s, 1))
val counts = pairs.reduceByKey((a, b) => a + b)
```
```python
lines = sc.textFile("data.txt")
pairs = lines.map(lambda s: (s, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)
```
#####Transformations
|Transformation|Meaning|
|--------------|-------|
|`map(func)`|Return a new distributed dataset formed by passing each element of the source through a function func.|
|`filter(func)`|Return a new dataset formed by selecting those elements of the source on which func returns true.|
|`flatMap(func)`|Similar to map, but each input item can be mapped to 0 or more output items (so func should return a Seq rather than a single item).|
|`mapPartitions(func)`|Similar to map, but runs separately on each partition (block) of the RDD, so func must be of type `Iterator<T> => Iterator<U>` when running on an RDD of type T.|
|`mapPartitionsWithIndex(func)`|Similar to mapPartitions, but also provides func with an integer value representing the index of the partition, so func must be of type `(Int, Iterator<T>) => Iterator<U>` when running on an RDD of type T.|
|`sample(withReplacement, fraction, seed)`|Sample a fraction fraction of the data, with or without replacement, using a given random number generator seed.|
|`union(otherDataset)`|Return a new dataset that contains the union of the elements in the source dataset and the argument.|
|`intersection(otherDataset)`|Return a new RDD that contains the intersection of elements in the source dataset and the argument.|
|`distinct([numPartitions]))`|Return a new dataset that contains the distinct elements of the source dataset.|
|`groupByKey([numPartitions])`|When called on a dataset of (K, V) pairs, returns a dataset of `(K, Iterable<V>) pairs. Note: If you are grouping in order to perform an aggregation (such as a sum or average) over each key, using reduceByKey or aggregateByKey will yield much better performance.Note: By default, the level of parallelism in the output depends on the number of partitions of the parent RDD. You can pass an optional numPartitions argument to set a different number of tasks.|
|`reduceByKey(func, [numPartitions])`|When called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function func, which must be of type (V,V) => V. Like in groupByKey, the number of reduce tasks is configurable through an optional second argument.|
|`aggregateByKey(zeroValue)(seqOp, combOp, [numPartitions])`|When called on a dataset of (K, V) pairs, returns a dataset of (K, U) pairs where the values for each key are aggregated using the given combine functions and a neutral "zero" value. Allows an aggregated value type that is different than the input value type, while avoiding unnecessary allocations. Like in groupByKey, the number of reduce tasks is configurable through an optional second argument.|
|`sortByKey([ascending], [numPartitions])`|When called on a dataset of (K, V) pairs where K implements Ordered, returns a dataset of (K, V) pairs sorted by keys in ascending or descending order, as specified in the boolean ascending argument.|
|`join(otherDataset, [numPartitions])`|When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key. Outer joins are supported through leftOuterJoin, rightOuterJoin, and fullOuterJoin.|
|`cogroup(otherDataset, [numPartitions])`|When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (Iterable<V>, Iterable<W>)) tuples. This operation is also called groupWith.|
|`cartesian(otherDataset)`|When called on datasets of types T and U, returns a dataset of (T, U) pairs (all pairs of elements).|
|`pipe(command, [envVars])`|Pipe each partition of the RDD through a shell command, e.g. a Perl or bash script. RDD elements are written to the process's stdin and lines output to its stdout are returned as an RDD of strings.|
|`coalesce(numPartitions)`|Decrease the number of partitions in the RDD to numPartitions. Useful for running operations more efficiently after filtering down a large dataset.|
|`repartition(numPartitions)`|Reshuffle the data in the RDD randomly to create either more or fewer partitions and balance it across them. This always shuffles all data over the network.|
|`repartitionAndSortWithinPartitions(partitioner)`|Repartition the RDD according to the given partitioner and, within each resulting partition, sort records by their keys. This is more efficient than calling repartition and then sorting within each partition because it can push the sorting down into the shuffle machinery.|
```scala
val dataRDD: RDD[Int] = sparkContext.makeRDD(List(1,2,3,4))

//将待处理的数据以分区为单位发送到计算节点进行处理，这里的处理是指可以进行任意的处 理
val dataRDD1: RDD[Int] = dataRDD.mapPartitions(
   datas => {
       datas.filter(_==2)
   }
)

val dataRDD1 = dataRDD.mapPartitionsWithIndex(
   (index, datas) => {
       datas.map(index, _)
   }
)

val dataRDD = sparkContext.makeRDD(List(
   List(1,2),List(3,4)
),1)
val dataRDD1 = dataRDD.flatMap(
   list => list
)

//一个组的数据在一个分区中，但是并不是说一个分区中只有一个组
val dataRDD = sparkContext.makeRDD(List(1,2,3,4),1)
val dataRDD1 = dataRDD.groupBy(
_%2 )

val dataRDD = sparkContext.makeRDD(List(
   1,2,3,4
),1)
val dataRDD1 = dataRDD.filter(_%2 == 0)

val dataRDD = sparkContext.makeRDD(List(
   1,2,3,4
),1)
// 抽取数据不放回(伯努利算法)
// 伯努利算法:又叫 0、1 分布。例如扔硬币，要么正面，要么反面。
// 具体实现:根据种子和随机算法算出一个数和第二个参数设置几率比较，小于第二个参数要，大于不 要
// 第一个参数:抽取的数据是否放回，false:不放回
// 第二个参数:抽取的几率，范围在[0,1]之间,0:全不取;1:全取;
// 第三个参数:随机数种子
val dataRDD1 = dataRDD.sample(false, 0.5)
// 抽取数据放回(泊松算法)
// 第一个参数:抽取的数据是否放回，true:放回;false:不放回
// 第二个参数:重复数据的几率，范围大于等于 0.表示每一个元素被期望抽取到的次数
// 第三个参数:随机数种子
val dataRDD2 = dataRDD.sample(true, 2)

val dataRDD = sparkContext.makeRDD(List(
   1,2,3,4,1,2
),2)
val dataRDD1 = dataRDD.sortBy(num=>num, false, 4)


val dataRDD1 = sparkContext.makeRDD(List(1,2,3,4))
val dataRDD2 = sparkContext.makeRDD(List(3,4,5,6))
val dataRDD = dataRDD1.intersection(dataRDD2)
val dataRDD = dataRDD1.union(dataRDD2)
val dataRDD = dataRDD1.subtract(dataRDD2)

//其中，键值对中的 Key 为第 1 个 RDD 中的元素，Value 为第 2 个 RDD 中的相同位置的元素。
val dataRDD = dataRDD1.zip(dataRDD2)
 

```

#####Actions
|Action|Meaning|
|------|-------|
|`reduce(func)`|Aggregate the elements of the dataset using a function func (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel.|
|`collect()`|Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data.|
|`count()`|Return the number of elements in the dataset.|
|`first()`|	Return the first element of the dataset (similar to take(1)).|
|`take(n)`	|Return an array with the first n elements of the dataset.|
|takeSample(withReplacement, num, [seed])|	Return an array with a random sample of num elements of the dataset, with or without replacement, optionally pre-specifying a random number generator seed.
|takeOrdered(n, [ordering])|	Return the first n elements of the RDD using either their natural order or a custom comparator.
|saveAsTextFile(path)|	Write the elements of the dataset as a text file (or set of text files) in a given directory in the local filesystem, HDFS or any other Hadoop-supported file system. Spark will call toString on each element to convert it to a line of text in the file.
|saveAsSequenceFile(path)(Java and Scala)|	Write the elements of the dataset as a Hadoop SequenceFile in a given path in the local filesystem, HDFS or any other Hadoop-supported file system. This is available on RDDs of key-value pairs that implement Hadoop's Writable interface. In Scala, it is also available on types that are implicitly convertible to Writable (Spark includes conversions for basic types like Int, Double, String, etc).
|saveAsObjectFile(path)(Java and Scala)|	Write the elements of the dataset in a simple format using Java serialization, which can then be loaded using SparkContext.objectFile().
|countByKey()|	Only available on RDDs of type (K, V). Returns a hashmap of (K, Int) pairs with the count of each key.
|foreach(func)|	Run a function func on each element of the dataset. This is usually done for side effects such as updating an Accumulator or interacting with external storage systems.

#####Shuffle operations
Certain operations within Spark trigger an event known as the shuffle. The shuffle is Spark’s mechanism for re-distributing data so that it’s grouped differently across partitions. This typically involves copying data across executors and machines, making the shuffle a complex and costly operation.

#####RDD Persistence
One of the most important capabilities in Spark is persisting (or caching) a dataset in memory across operations. When you persist an RDD, each node stores any partitions of it that it computes in memory and reuses them in other actions on that dataset (or datasets derived from it). This allows future actions to be much faster (often by more than 10x). Caching is a key tool for iterative algorithms and fast interactive use.

You can mark an RDD to be persisted using the persist() or cache() methods on it. The first time it is computed in an action, it will be kept in memory on the nodes. Spark’s cache is fault-tolerant – if any partition of an RDD is lost, it will automatically be recomputed using the transformations that originally created it.

#####Shared Variables
Normally, when a function passed to a Spark operation (such as map or reduce) is executed on a remote cluster node, it works on separate copies of all the variables used in the function. These variables are copied to each machine, and no updates to the variables on the remote machine are propagated back to the driver program. Supporting general, read-write shared variables across tasks would be inefficient. However, Spark does provide two limited types of shared variables for two common usage patterns: broadcast variables and accumulators.

######Broadcast Variables
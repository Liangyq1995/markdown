###### 常用的参数配置

```python
sparkSession = SparkSession.builder.appName("liangyongqi_test") \
  .config("spark.blacklist.enabled", 'false')\
  .config("spark.dynamicAllocation.maxExecutors", '500')\
  .config("spark.sql.shuffle.partitions", '600')\
  .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class","DockerLinuxContainer") \
  .config("spark.executorEnv.yarn.nodemanager.container-executor.class","DockerLinuxContainer") \
  .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name","bdp-docker.jd.com:5000/wise_mart_bag:latest")  \
  .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name","bdp-docker.jd.com:5000/wise_mart_bag:latest")  \
  .config("spark.sql.viewPermission.enabled", 'true')\
  .config("spark.sql.parser.quotedRegexColumnNames", 'false')
  .enableHiveSupport().getOrCreate()
sc = sparkSession.sparkContext
spark = sparkSession
```

###### 使用视图

```shell
--conf spark.sql.viewPermission.enabled=true
--conf spark.sql.parser.quotedRegexColumnNames=false
```

###### String Split

```
关于点的问题是用string.split("[.]") 解决。

关于竖线的问题用 string.split("\\|")解决。

关于星号的问题用 string.split("\\*")解决。

关于斜线的问题用 sring.split("\\\\")解决。

关于中括号的问题用 sring.split("\\[\\]")解决。
```

Spark 3 has new array functions that make working with ArrayType columns much easier. The `transform` and `aggregate` array functions are especially powerful general purpose functions. They provide functionality that’s equivalent to `map` and `fold` in Scala and make it a lot easier to work with ArrayType columns.

###### xgboost

```scala
//integrate it with other single node libraries for further processing.
//保存到hdfs上
val path = "/user/jd_ad/ads_polaris/liangyongqi8/jinxizhushou/egg_open_model.bin"
val fsConfig = new Configuration()
val dfs = FileSystem.get(fsConfig)
val dfsPath = new Path(path)
val saveModelPath = dfs.create(dfsPath)
model.nativeBooster.saveModel(saveModelPath)
```

###### `DataFrame` to `HashMap`

```scala
import scala.collection.mutable

val hashMap = data.select("area_code", "area_name").rdd
				  .map(row=>row.getAs("area_code").toString -> row.getAs("area_name").toString).collectAsMap().asInstanceOf[mutable.HashMap[String, String]]
```

###### `DataFrame`操作

```scala
// 重命名多列, 个性化单独修改
val renameMap = Map("_1" -> "foo", "_3" -> "bar")
df.select(df.columns.map(c => col(c).as(renameMap.getOrElse(c, c))): _*)

//基于现有的列生成多列
val processSeq = Seq("A", "B", "C", "SUM")
processSeq.foldLeft(df)((df, c) =>
  df.withColumn(s"ratio_$c",  col(c) / col("sum"))
)

//对现有的多列进行多个操作

def gen_expr(feature_array: Array[String]): Seq[Column] ={
        val expr = feature_array.toSeq.filter(x=>x!="device_id" && x!="hourGap").map(x=>
            collect_list(concat(col("hourGap"),lit(":"),col(x))).alias(x+"_hourGap_list")
        )
        expr
    }
// df进行调用，注意agg括号中的写法
val expr = gen_expr(df.columns)
val df_new = df.groupBy("device_id").agg(expr.head, expr.tail:_*)

```

```python
spark.sql(
      """
         CREATE
         	EXTERNAL TABLE IF NOT EXISTS tmp.tmp_dm_user_rec_batch_merge_seq_egg_open
         	(
         		user_log_acct STRING,
         		userviewskucountm10 INT,
         		userviewskucountm30 INT,
         		view_sku_seq array<float>,
         		view_brand_seq array<float>
         	)
         	COMMENT '序列特征测试' PARTITIONED BY
         	(
         		dt STRING 
         	)
         	STORED AS ORC LOCATION 'hdfs://ns1007/user/mart_sch/tmp.db/tmp_dm_user_rec_batch_merge_seq_egg_open'
         """)
         
test.withColumn("view_sku_seq_arr", F.split("view_sku_seq", ",").cast("array<float>")).withColumn("view_brand_seq_arr", F.split("view_brand_seq", ",").cast("array<float>"))
```

```hive
SET mapred.job.priority = HIGH;  -- 提高任务优先级 
SET hive.exec.dynamic.partition = true;  -- 打开动态分区 ( 将A表的某个字段作为 partition 字段, 插入到B表中 , 默认为 false )
SET hive.exec.dynamic.partition.mode = nonstrict;  --( 与上述配合使用, 默认为: strict , 设置为 nonstrict 即为开启全部动态分区  )
SET hive.exec.max.dynamic.partitions = 100000;  -- ( 设置动态分区的最大限制  )
SET hive.exec.max.dynamic.partitions.pernode = 100000; 
SET hive.exec.parallel = true;  --( 开启并发 )
SET hive.exec.parallel.thread.number = 10;  --( 并发线程数 )
SET hive.input.format = org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
SET hive.hadoop.supports.splittable.combineinputformat = true;
SET mapreduce.input.fileinputformat.split.maxsize = 256000000;
SET mapreduce.input.fileinputformat.split.minsize.per.node = 256000000;
SET mapreduce.input.fileinputformat.split.minsize.per.rack = 256000000;
SET hive.merge.mapfiles = true;   -- ( True时会合并map输出 )
SET hive.merge.mapredfiles = true; -- ( 在Map-Reduce的任务结束时合并小文件 )
SET hive.merge.size.per.task = 256000000;  -- ( 合并操作后的单个文件大小 )
SET hive.merge.smallfiles.avgsize = 256000000;  -- ( 输出文件平均大小设定值时，小于该值启动合并操作 ) 
SET hive.optimize.sort.dynamic.partition = true; 
```

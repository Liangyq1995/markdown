##### null值问题

```python
spark.sql("""
select true or null
""") # 返回true

spark.sql("""
select true and null
""") # 返回 null
```

`filter`或`where`时，会把判断为`null`的过滤掉，`null`和其他比较返回`null`除了

```python
spark.sql("""
select null is not null
""") #False
spark.sql("""
select null is null
""") 
```

spark函数操作列时候，对于无法操作的，一般返回`null`，字符串操作的一些函数可能会返回`''`

分组`groupby`时，null会作为单独一列进行统计。



##### 过滤

`filter`和`where`保留满足条件数据，会把返回`false`和`null`的都过滤掉。

```python
df.filter(df.age > 3) #保留condition为True的，去除为false和null
```

###### `having`

```python
sqlContext.sql("""
select Category,count(*) as 
count from hadoopexam where HadoopExamFee<3200  
group by Category having count>10
""")
#等价于
df.filter(df.HadoopExamFee<3200)
  .groupBy('Category')
  .agg(count('Category').alias('count'))
  .filter(col('count')>10)
```

#####小表大表关联优化
优化方案1:调整大小表位置,将小表放在左边后，提升至29s (该方案一直不太明白为啥会提升，执行计划里显示的也就是大小表位置调换下而已，跟之前的没其他区别)
优化方案2: 将 or 改成 union,提升至35s(各种调整,一直怀疑跟or有关系,后面调整成union其他不变,果真效率不一样;但方案1只是调整了下大小表顺序,并未调整其他,其效率同样提升很大;不太明白sparksql内部到底走了什么优化机制,后面继续研究);
优化方案3： 采用cache+broadcast方式,提升至20s（该方案将小表缓存至内存，进行map侧关联），原理：使用broadcast将会把小表分发到每台执行节点上，因此，关联操作都在本地完成，基本就取消了shuffle的过程，运行效率大幅度提高。
```sql
cache table cta
    as
            SELECT
                    round(sum(click) / sum(imp), 4) avg_click_rate
            FROM
                    schema.srctable1
            WHERE
                    date = '20171020';
```

#####用pyspark预测Tensorflow
```shell
 #!/usr/bin/env bash
export PYSPARK_PYTHON=./clv_py_env/python36/bin/python && \         #注：指定worker端python环境地址
export PYSPARK_DRIVER_PYTHON=/usr/local/anaconda2/envs/python36/bin/python && \   #注：本地driver端的python环境地址，最好跟executor端的版本、包一致
spark-submit \
        --master yarn \
        --queue root.bdp_jmart_ad_data.jd_ad_data_mo \
        --archives hdfs://ns1018/user/jd_ad/jd_ad_ads_mo/pyenvs/python36.zip#clv_py_env \              #注：指定worker端的python环境，#后面是起个别名,别名对应上面worker端的python工作环境
        --conf spark.yarn.maxAppAttempts=3 \
        --conf spark.task.cpus=4 \
        --conf spark.executor.memoryOverhead=30G \
        --driver-memory 20g \
        --driver-cores 6 \
        --executor-memory 20g \
        --executor-cores 6 \
        --num-executors 300 \
        --py-files zero_inflated_lognormal.py,clv_lib.py \
        --files model_1320.h5,1320_min_max_scaler.bin \
        "$@"
             
```
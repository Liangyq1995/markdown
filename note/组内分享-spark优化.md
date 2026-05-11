公司的spark配置中spark.dynamicAllocation.enabled=True，默认启用启用动态分配，在executor空闲spark.dynamicAllocation.executorIdleTimeout(默认60s)之后将被释放。
--num-executors 80 在0～100之间或者不设置的话，spark.dynamicAllocation.minExecutors=2, spark.dynamicAllocation.maxExecutors=100, 
spark.dynamicAllocation.initialExecutors=2
不产生影响，如果>100,则spark.dynamicAllocation.maxExecutors=num_executors, 其他不产生影响。默认开启spark.dynamicAllocation.enabled=True

最大并发：spark.dynamicAllocation.maxExecutors * executor-cores/spark.task.cpus, task.cpus默认为1，建议不要自行调整。

spark.executor.memoryOverhead默认为3G




Spark Application的main方法（SparkContext相关的代码）运行在Driver上，当用于计算的RDD触发Action动作之后，会提交Job，那么RDD就会向前追溯每一个transformation操作，直到初始的RDD开始，这之间的代码运行在Executor。

driver做什么

运行应用程序的main函数

创建spark的上下文

划分RDD并生成有向无环图（DAGScheduler）

与spark中的其他组进行协调，协调资源等等（SchedulerBackend）

生成并发送task到executor（taskScheduler）

History首页会加载所有的application，所以加载会比较慢，对你来说不需要等页面加载出来
你可以直接拼接URL：
这样拼接：http://history server（以上2.4~3.0的域名）/history[固定不变]/applicationID/[1/如果是cluster模式需要添加]stages[固定不变]/
 eg: hope集群，spark 2版本，client模式: http://hope.sparkhs-2.jd.com/history/application_3717273103802_13400237/stages/
10k集群，spark 3版本，cluster模式: http://10k.sparkhs-3.jd.com/history/application_3717273103802_13400237/1/stages/


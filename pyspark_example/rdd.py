#rdd 编程
#创建rdd
#常用action操作
#常用transformation
#常用pairRDD的转换操作
#缓存操作
#共享变量
#分区操作

#map
#flatMap
#mapPartitions
#filter
#count
#reduce
#take
#saveAsTextFile
#collect
#join
#union
#persist
#repartition
#reduceByKey
#aggregateByKey
from pyspark.sql import SparkSession

sc=SparkSession.builder.appName("test").config("master","local[4]").enableHiveSupport().getOrCreate()
file="./data/hello.txt"
rdd=sc.textFile(file,3)
rdd.collect()


rdd=sc.parallelize(range(1,11),2)
rdd.collect()


#常用action操作

rdd=sc.parallelize(range(10),5)
all_data=rdd.collect()


#take

rdd=sc.parallelize(range(10),5)
part_data=rdd.take(4)


#takeSample

rdd=sc.parallelize(range(10),5)
sample_data=rdd.takeSample(False,10,0)


#first
rdd=sc.parallelize(range(10),5)
data_count=rdd.count()

#count
rdd=sc.parallelize(range(10),5)
data_count=rdd.count()

#reduce

rdd=sc.parallelize(range(10),5)
data_count=rdd.reduce(lambda x,y:x+y)
print(data_count)

#foreach
rdd=sc.parallelize(range(10),5)
accum=sc.accumulator(0)
rdd.foreach(lambda x:accum.add(x))
print(accum.value)

#countByKey rdd按照key统计数据
pairRdd=sc.parallelize([(1,1),(2,2),(3,9)])
pairRdd.countBykey()


#saveAsTextFile

text_file="./data/rdd.txt"
rdd=sc.parallelize(range(5))
rdd.saveAsTextFile(file)

#重新读入会被解析文本
rdd_loaded=sc.textFile(file)
rdd_loaded.collect()


#常用transformation操作

#map

rdd=sc.parallelize(range(10),3)
rdd.collect()
rdd.map(lambda x:x**2).collect()

#filter
rdd=sc.parallelize(range(10),3)
rdd.filter(lambda x:x>5).collect()

#flatMap
rdd=sc.parallelize(["hello world","hello china"])
rdd.map(lambda x:x.split(" ")).collect()
rdd.flatMap(lambda x:x.split(" ")).collect()

#sample
rdd=sc.parallelize(range(10),1)
rdd.sample(False,0.5,0)

#distinct
rdd=sc.parallelize([1,1,2,2,3,3,4,5])
rdd.distinct().collect()

#subtract
a=sc.parallelize(range(10))
b=sc.parallelize(range(5,15))
a.subtract(b).collect()

#union

a=sc.parallelize(range(5))
b=sc.parallelize(range(3,8))
a.union(b).collect()

#intersection
a=sc.parallelize(range(1,6))
b=sc.parallelize(range(3,9))
a.intersection(b).collect()

#cartesian

boys=sc.parallelize(["lilei","tom"])
girls=sc.parallelize(["hanmeimei","lily"])
boys.cartesian(girls).collect()


#sortBy

rdd=sc.parallelize([(1,2,3),(3,2,2)])
rdd.sortBy(lambda x:x[2]).collect()

#zip
rdd_name=sc.parallelize(["lilei","hanmeimei","lily"])
rdd_age=sc.parallelize([19,18,20])
rdd_zip=rdd_name.zip(rdd_age)
print(rdd_zip.collect())

#zipWithIdex
rdd_name=sc.parallelize(["lilei","hanmeimei","lily"])
rdd_index=rdd_name.zipWithIndex()
print(rdd_index.collect())

#常用pairRDD的转换操作
#reduceByKey
rdd=sc.parallelize([("hello",1),("world",2)])
rdd.reduceByKey(lambda x,y:x+y).collect()

#groupByKey
rdd=sc.parallelize([("hello",1),("world",2)])
rdd.groupByKey().collect()

#sortByKey
rdd=sc.parallelize([("hello",1),("world",2),("hello",3),("world",5)])
rdd.sortByKey().collect()

#join
age=sc.parallelize([("lilei",18),("hanmeimei",16),("jim",20)])
gender=sc.parallelize([("lilei","male"),("hammeimei","female"),("lucy","female")])
age.join(gender).collect()

#leftOuterJoin 和rightOuterJoin

age=sc.parallelize([("lilei",18)])
gender=sc.parallelize([("hanmeimei","male")])
age.leftOuterJoin(gender).collect()


age=sc.parallelize(["lilei",18])
gender=sc.parallelize(["lelei","male"])
age.rightOuterJoin(gender).collect()


#cogroup
result=x.cogroup(y).collect()

#subtractByKey
x=sc.parallelize([("a",1),("b",2)])
y=sc.parallelize([("a",2),("b",(1,2))])
x.subtractByKey(y).collect()

#foldByKey
x=sc.parallelize([("a",1),("b",2),("c",3)])
x.foldByKey(1,lambda x,y:x*y).collect()

#缓存操作

#如果一个rdd被多个任务用作中间量，对cache进行缓存到内存中对加快运算
#对一个rdd进行cache后，该rdd不会被立即缓存，而是等到第一次被计算出来时才进行缓存
#可以使用persist明确指定存储级别，memory_only memory_and_disk
#如果一个rdd后面不再用到，可以用unpersist释放缓存，unpersist是立即执行的
#缓存数据不会切断血缘依赖关系，这是因为缓存数据某些分区所在的节点有可能有故障，例如内存溢出或者节点损坏
#这时候根据血缘关系重新计算这个分区的数据



#cache缓存在内存中，使用存储级别memory_only memory_only 意味着如果内存处处不下，放弃存储其余部分，需要时重新计算
a=sc.parallelize(range(10000),5)
a.cache()
sum_a=a.reduce(lambda x,y:x+y)
cnt_a=a.count()
mean_a=sum_a/cnt_a
print(mean_a)

#persist缓存到内存或者磁盘中，默认使用存储级别memory_and_disk
#memory_and_disk 意味着如果内存存储不下，其余部分存储到磁盘中
#persist可以指定其他存储级别，cache相当于persist(memory_only)

from pyspark.storagelevel import StorageLevel
a=sc.parallelize(range(10000),5)
a.persist(StorageLevel.MEMORY_AND_DISK)
sum_a=a.reduce(lambda x,y:x+y)
cnt_a=a.count()
mean_a=sum_a/cnt_a
a.unpersit()
print(mean_a)

#共享变量
# 广播变量和累加器
#
# 广播变量是不可变变量，实现不同节点不同任务之间的变量
# 广播变量在每个机器上缓存一个只读的变量，而不是为每个task生成一个副本，可以减少数据的传输
#
# 累加器主要是不同节点和driver之间共享变量，只能实现计数或者累加功能
# 累加器的值只有在driver上是可读的，在节点上不可见

broads=sc.broadcast(100)
rdd=sc.parallelize(range(10))
print(rdd.map(lambda x:x+broads.value)).collect()
print(broads.value)

#累加器 只能在driver上可读，在其它节点只能进行累加
total=sc.accumulator(0)
rdd=sc.parallelize(range(10),3)
rdd.foreach(lambda x:total.add(x))
total.value

#计算数据的平均值
rdd=sc.parallelize([1,2,3,4])
total=sc.accumulator(0.1)
count=sc.accumulator(0)

def func(x):
    total.add(x)
    count.add(1)

rdd.foreach(func)
total.value/count.value

#分区操作
#分区操作包括改变分区操作，以及针对分区执行转换操作
#glom 将一个分区内的数据转换为一个列表作为一行
#coalesce:shuffle可选，默认为False情况下窄依赖，不能增加分区 repartition和partitionBy 调用它实现
#repartition 按随机数进行shuffle,相同key不一定在同一个分区
#partitionBy 按照key进行shuffle,相同key放入同一个分区
#hashpartitioner 默认分区器，根据key的hash值进行分区，相同key进入同一个分区，效率较高，key不可为array
#rangepartitioner 只在排序相关函数中使用，除相同的key进入同一分区，相邻的key也会进入同一分区，key必须可排序
#taskContext 获取当前分区id方法 taskContext.get.partitonId
#mapPartitions 每次处理分区内的一批数据，适合需要分批处理数据的情况，比如将数据插入某个表，每批数据只需要开启一次数据库连接，大大减少了连接开支
#mapPartitonswithidx 类似mappartions 提供了分区索引，输入参数为(i,Iterator)
#foreachPartition 类似foreach 但每次提供一个partition的一批数据


















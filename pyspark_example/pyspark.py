
spark_home="/Users/chenjing22/ProgramFiles/spark-3.0.1-bin-hadoop3.2"
python_path="/Users/liangyun/anaconda3/bin/python"

from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession,SQLContext,HiveContext

sc=SparkSession.builder\
    .appName("test")\
    .config("master","local[4]")\
    .enableHiveSupport()\
    .getOrCreate()



#求平均数
data=[1,5,7,10,23,20]
dfdata=sc.createDataFrame([(x,) for x in data]).toDF("value")
dfagg=dfdata.agg({"value":"avg"})
dfagg.show()


#从本地文件系统中加载数据
file='./data/hello.txt'
rdd=sc.textFile(file,3)
rdd.collect()

#parallize将driver中的数据结构生成rdd 第二个参数为指定分区的rdd
rdd=sc.parallelize(range(1,11),2)
rdd.collect()

#常用action操作
#action操作将触发基于rdd依赖关系的计算
#collect

rdd=sc.parallelize(range(10),5)
all_data=rdd.collect()


#take
rdd=sc.parallelize(range(10),5)
part_data=rdd.take(4)
part_data

rdd=sc.parallelize(range(10),5)
sample_data=rdd.takeSample(False,10,0)
sample_data

#first取第一个数据
rdd=sc.parallelize(range(10),5)
first_data=rdd.first()
print(first_data)

#
rdd=sc.parallelize(range(10),5)
data_count=rdd.count()
print(data_count)

#reduce
rdd=sc.parallelize(range(10),5)
rdd.reduce(lambda x,y:x+y)

#foreach
rdd=sc.parallelize(range(10),5)
accum=sc.accumulator(0)
rdd.foreach(lambda x:accum.add(x))

pairRdd=sc.parallelize([(1,1),(1,4),(3,9),(2,16)])
pairRdd.countBykey()

text_file="./data/rdd.txt"
rdd=sc.parallelize(range(5))
rdd.saveAsTextFile(text_file)

#常用transformation操作
rdd_loaded=sc.textFile(file)
rdd_loaded.collect()

rdd=sc.parallelize(range(10),3)
rdd.collect()

rdd.map(lambda x:x**2).collect()

rdd= sc.parallelize(range(10),3)
rdd.filter(lambda x:x>5).collect()

rdd=sc.parallelize(['sffdf','fsdaf'])
rdd.map(lambda x:x.split(" ")).collect()

rdd.flatMap(lambda x:x.split(" ")).collect()

rdd.sample(False,0.5,0).collect()

rdd.distinct().collect()

#subtract
a=sc.parallelize(range(10))
b=sc.parallelize(range(5,15))
a.subtract(b).collect()

#union
a= sc.parallelize(range(5))
b=sc.parallelize(range(3,8))
a.union(b).collect()

#intersection
a=sc.parallelize(range(1,6))
b=sc.parallelize(range(3,9))




from pyspark.sql import functions as F
data=[1,5,7,9,10]
dfdata=sc.createDataFrame([(x,1) for x in data]).toDF("key")
dfdata.groupby("key").agg(F.count("value").alias("count")).cache()
max_count=dfdata.agg(F.max("count").alias("max_count")).take(1)[0]["max_count"]
dfmode=dfdata.where("count={}".format(max_count))
mode=dfmode.get(F.expr("mean(key) as mode"))
print("mode:",mode)

dfdata.unpersist()


students=[{"lilei",18,87},("HanMeiMei",16,77),("dachui",16,66)]
n=3

dfstudents=sc.createDataFrame(students).toDF("name","age","score")
dftopn=dfstudents.orderBy("score",ascending=False).limit(n)

dftopn.show()
data=[1,7,8,5]

#排序并返回序号
from copy import deepcopy
from pyspark.sql import types as f
from pyspark.sql import Row,DataFrame

field_name=""
def addLongIndex(df,filed_name):
    schema=deepcopy(df.shema)
    schema=schema.add(f.structField(field_name,f.LongType()))
    rdd_with_index=df.rdd.zipWithIndex()

    def merge_row(t):
        row,index=t
        dic=row.asDict()
        dic.update({field_name:index})
        row_merge=Row(**dict)
        return row_merge

    rdd_row=rdd_with_index.map(lambda t:merge_row(t))

    return sc.createDataFrame(rdd_row,schema)


dfdata=sc.createDataFrame([(x,) for x in data]).toDF("value")
dfsorted=dfdata.sort(dfdata["value"])

dfsorted_index=addLongIndex(dfsorted,"index")


#二次排序
students=[{"lilei",18,87},{"dachui",16,66}]
dfstudents=spark.createDataFrame(students).toDF("name","age","score")
dfsorted=dfstudents.orderBy(dfstudents["score"].desc(),dfstudents["age"].desc())
dfsorted.show()

#连接操作

from pyspark.sql import functions as F
classes=[{"class1","lilei"},("class1","hanmeimei")]
scores=[("lilei",76),("hanmeimei",80)]

dfclass=sc.createDataFrame(classes).toDF("class","name")
dfscore=sc.createDataFrame(scores).toDF("name","score")

dfstudents=dfclass.join(dfscore,on="name",how="left")

dfagg=dfstudents.groupby("class").agg(F.avg("score").alias("avg_score").where("avg_score>75.0"))



#分组求众数

from pyspark.sql import functions as F

def mode(arr):
    dict_cnt={}
    for x in arr:
        dict_cnt[x]=dict_cnt.get(x,0)+1
    max_cnt=max(dict_cnt.values())
    most_values=[k for k,v in dict_cnt.items() if v==max_cnt]
    s=0.0
    for x in most_values:
        s=s+x
    return s/len(most_values)

sc.udf.register("udf_mode",mode)
sfstudents=sc.createDataFrame(students).toDF("class","score")
dfscores=dfstudents.groupby("class").agg(F.collect_list("score").alias("scores"))
dfmode=dfscores.selectExpr("class","udf_mode(scores) as mode_score")
dfmode.show()







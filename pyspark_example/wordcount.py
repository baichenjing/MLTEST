##rdd api
import random

from jieba import xrange

#wordcount
from pyspark_example import SparkContext as sc, Row
from pyspark_example.ml.classification import LogisticRegression
from pyspark_example.shell import sqlContext

text_file=sc.textFile("hdfs://")
counts=text_file.flatMap(lambda line:line.split(" "))\
    .map(lambda word:(word,1)).\
    reduceByKey(lambda a,b:a+b)

counts.saveAsTextFile("hdfs://")


# pi estimation
def inside(p):
    x,y=random.random(),random.random()
    return x*x+y*y<1

count=sc.parallelize(xrange(0,NUM_SAMPLES))\
    .filter(inside).count()
print("pi is roughly %f" %(4.0*count/NUM_SAMPLES))

# Dataframe api examples

textFile=sc.textFile("hdfs://")
df=textFile.map(lambda r:Row(r)).toDF(["line"])
errors=df.filter(col("line")).like("%error%")
errors.count()

errors.filter(col('line').like("%mysql%")).count()

errors.filter(col('line').like('%mysql%')).collect()

#simple data operations
url="jdbc:mysql://yourip.yourport/test?user=yourname;password=yourpassword"
df=sqlContext.read.format("jdbc").option('url',url).option('dbtable','people').load()
df.printSchema()

countsByAge=df.groupBy('age').count()
countsByAge.show()

countsByAge.write.format('json').save('s3a://...')

#prediction with logistic regression

df=sqlContext.createDataFrame(data,['lable','reatures'])
lr=LogisticRegression(maxIter=10)
model=lr.fit(df)
model.transform(df).show()



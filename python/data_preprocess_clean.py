import pandas as pd
import numpy as np

fileNameStr='./actual transactions'
DataDF=pd.read_csv(fileNameStr,encoding="iso-8859-1",dtype=str)
DataDF=pd.read_csv(fileNameStr,encoding="utf-8")
DataDF=pd.read_csv(fileNameStr,encoding="utf-8",dtype=str)

DataDF.info()
DataDF.dtypes

DataDF.shape

DataDF.isnull().sum().sort_values(ascending=False)
DataDF.head()
DataDF['Description'].unique()

np.set_printoptions(threshold=np.inf)

##更改列名
colNameDict={'InvolceDate':'saleDate','StockCode':'StockNo'}
salesDf.rename(columns=colNameDict,inplace=True)

#更改某一列数据类型
DataDF.loc[:,'InvoiceDate']=pd.to_datetime(DataDF.loc[:,'InvoiceDate'],format='%d/%m/%y',errors='coerce')

#选择子集
subDataDF1=DataDF["InvoiceDate"]
subDataDF1=DataDF[["InvoiceDate","UnitPrice"]]

#利用切片筛选数据功能 df.loc
subDataDF1=DataDF,loc[:,"InvoiceDate"]
subDataDF1

#使用loc可以定位
subDataDF2=DataDF.loc[0:9,:]
subDataDF2

subDataDF3=DataDF.loc[1:9,"StockCode":"CustomerID"]
DataDF.loc[:,'UnitPrice']>0

querySer=DataDF.loc[:,'Qunantity']>0
#删除异常值前
DataDF=DataDF.loc[querySer,:]
#删除异常值后

1.DataDF['Description']=DataDF['Description'].str.upper()
2.去除字符串符号 去乱码
3.空格分隔

def splitSaletime(timeColSer):
    timeList=[]
    for value in timeColSer:
        dateStr=value.split(' ')[0]
        timeList.append(dateStr)

    timeSer=pd.Series(timeList)
    return timeSer

DataDF.loc[:,'InvoiceDate']=splitSaletime(DataDF.loc[:,'InvoiceDate'])

#缺失值有三种
1.python 内置none值
2.在pandas中,将缺失值表示为NA,表示不可用not available
3.对于缺失值数据,pandas使用浮点值Nan表示缺失数据。后面出来数据，如果遇到错误，说什么float错误，那就是缺失值，需要处理掉

所以缺失值有3种，None,NA,NaN

None是python的一种数据类型
NaN是浮点类型
两个都用作空值
from numpy import NaN

#去除缺失值
DataDF.isnull().sum().sort_values(ascending=False)
DataDF.dropna(how='any')
DataDF.dropna(how='all')
DataFrame.dropna(axis=0,how='any',thresh=None,subset=None,inplace=False)
DataDF.dropna(thresh=6)

#填充缺失内容
1)以业务知识或经验推测填充缺失值
2）以同一指标的计算结果填充缺失值
3）用相邻值填充缺失值
4)以不同指标的计算结果填充缺失值

去除缺失值的知识点
1）用默认值填充 df.fillna('not given')
2) df.fillna(df.mean())
3)df.unitprice.fillna(method='ffill')
4)以不同指标的计算结果填充缺失值







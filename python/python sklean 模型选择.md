1.主要功能如下
#####分类
#####回归
#####聚类
#####降维
#####模型选择
#####预处理

主要模块分类
##### sklearn.base  base classes and utility function
##### sklean.cluster clustering 聚类
##### sklean.cluster.bicluster 双向聚类
##### sklearn.covariance covariance estimators 协方差估计
##### sklearn.model_selection model selection 模型选择
##### sklearn.datasets datasets 数据集
##### sklearn.decomposition  matrix decomposition 矩阵分解
##### sklearn.dummy dummy estimator 虚拟估计
##### sklean.ensemble 集成方法
##### sklearn.exception 异常和警告
##### sklearn.feature_extraction 特征抽取
##### sklearn.featurn_selection 特征选择
##### sklearn.gaussian_process 高斯过程
##### sklearn.isotonic 保序回归
##### sklean.kernel_approximation 核逼近
##### sklean.kernel_ridge 岭回归
##### sklean.discriminant_analysis 判别分析
##### sklean.linear_model 广义线性模型
##### sklean.manifold 流形学习
##### sklean.metrics 度量 权值
##### sklearn.mixture 高斯混合模型
##### sklearn.pipeline 管道
##### sklearn.svm 支持向量机
##### sklearn.tree 决策树
##### skearn.utils 实用工具

数据预处理
#####from sklean import preprocessing
####将数据转换为标准正态分布 均值为0 方差为1
##### preprocessing.scale(X,axis=0,with_mean=True,copy=True)
####将数据在缩放在固定区间，默认缩放到区间[0,1]
#####preprocessing.minmax_scale(X,feature_range=(0,1),axis=0,copy=True)
标准化正太分布类
基于mean和std的标准化



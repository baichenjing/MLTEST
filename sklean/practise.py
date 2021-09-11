# coding=gbk

import time
from sklearn import metrics
import pickle as pickle
import pandas as pd


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    '''
    SVC参数解释?
（1）C:?目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default?C?=?1.0；?
（2）kernel：参数选择有RBF,?Linear,?Poly,?Sigmoid,?默认的是"RBF";?
（3）degree：if?you?choose?'Poly'?in?param?2,?this?is?effective,?degree决定了多项式的最高次幂；?
（4）gamma：核函数的系数('Poly',?'RBF'?and?'Sigmoid'),?默认是gamma?=?1?/?n_features;?
（5）coef0：核函数中的独立项，'RBF'?and?'Poly'有效；?
（6）probablity:?可能性估计是否使用(true?or?false)；?
（7）shrinking：是否进行启发式；?
（8）tol（default?=?1e?-?3）:?svm结束标准的精度;?
（9）cache_size:?制定训练所需要的内存（以MB为单位）；?
（10）class_weight:?每个类所占据的权重，不同的类设置不同的惩罚参数C,?缺省的话自适应；?
（11）verbose:?跟多线程有关，不大明白啥意思具体；?
（12）max_iter:?最大迭代次数，default?=?1，?if?max_iter?=?-1,?no?limited;?
（13）decision_function_shape?：?‘ovo’?一对一,?‘ovr’?多对多??or?None?无,?default=None?
（14）random_state?：用于概率估计的数据重排时的伪随机数生成器的种子。?

提示：7,8,9一般不考虑。?

def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
 
C,错误项的惩罚系数，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。
kernel，算法中采用的核函数类型
degree，这个参数只对多项式核函数有用，是指多项式核函数的阶数n
gamma，核函数系数，默认为auto
coef0，核函数中的独立项
probability，是否启用概率估计
shrinking，是否采用启发式收缩方式
tol，svm停止训练的误差精度
cache_size，指定训练所需要的内存，以MB为单位，默认为200MB。
class_weight，给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.
verbose，是否启用详细输出。此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
max_iter，最大迭代次数，如果为-1，表示不限制
random_state，伪随机数发生器的种子,在混洗数据时用于概率估计。
属性：
    svc.n_support_：各类各有多少个支持向量
    svc.support_：各类的支持向量在训练样本中的索引
    svc.support_vectors_：各类所有的支持向量
    '''
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(data_file):
    data = pd.read_csv(data_file)
    train = data[:int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)
    return train_x, train_y, test_x, test_y


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#线性判别分析
lda = LinearDiscriminantAnalysis()
'''
__init__函数
    def __init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver

solver="svd",求解算法，svd表示使用奇异值分解求解，不用计算协方差矩阵。
             lsqr表示最小平方qr分解
             eigen表示特征值分解
shrinkage=None,是否使用参数收缩
priors=None,用于LDA中贝叶斯规则的先验概率
components,需要保留的特征个数，小于等于n-1
store_covariance，是否计算每个类的协方差矩阵，0.19版本删除
用法：
    lda.fit(X_train, y_train)
属性：
    covariances_：每个类的协方差矩阵， shape = [n_features, n_features]
    means_：类均值，shape = [n_classes, n_features]
    priors_：归一化的先验概率
    rotations_：LDA分析得到的主轴，shape [n_features, n_component]
    scalings_：数组列表，每个高斯分布的方差σ
'''

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
'''
__init__函数
def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8):

hidden_layer_sizes，长度=n_layers-2, 默认(100，），第i个元素表示第i个隐藏层的神经元的个数。
activation，激活函数，默认为relu
solver，默认 ‘adam’，用来优化权重 
alpha，可选的，默认0.0001,正则化项参数 
batch_size，默认‘auto’,随机优化的minibatches的大小
learning_rate，默认‘constant’，用于权重更新
max_iter，默认200，最大迭代次数。 
random_state，可选，默认None，随机数生成器的状态或种子
shuffle，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。
tol，可选，默认1e-4，优化的容忍度 
learning_rate_int，默认0.001，初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。 
power_t，只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。 
verbose，是否将过程打印到stdout
warm_start，当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。 

属性：
    - classes_:每个输出的类标签 
    - loss_:损失函数计算出来的当前损失值 
    - coefs_:列表中的第i个元素表示i层的权重矩阵 
    - intercepts_:列表中第i个元素代表i+1层的偏差向量 
    - n_iter_ ：迭代次数 
    - n_layers_:层数 
    - n_outputs_:输出的个数 
    - out_activation_:输出激活函数的名称。
用法：
    - fit(X,y):拟合 
    - get_params([deep]):获取参数 
    - predict(X):使用MLP进行预测 
    - predic_log_proba(X):返回对数概率估计 
    - predic_proba(X)：概率估计 
    - score(X,y[,sample_weight]):返回给定测试数据和标签上的平均准确度 
    -set_params(**params):设置参数。
'''

if __name__ == '__main__':
    data_file = "H:\\Research\\data\\trainCG.csv"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
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
    SVC��������?
��1��C:?Ŀ�꺯���ĳͷ�ϵ��C������ƽ�������margin�ʹ�������ģ�default?C?=?1.0��?
��2��kernel������ѡ����RBF,?Linear,?Poly,?Sigmoid,?Ĭ�ϵ���"RBF";?
��3��degree��if?you?choose?'Poly'?in?param?2,?this?is?effective,?degree�����˶���ʽ����ߴ��ݣ�?
��4��gamma���˺�����ϵ��('Poly',?'RBF'?and?'Sigmoid'),?Ĭ����gamma?=?1?/?n_features;?
��5��coef0���˺����еĶ����'RBF'?and?'Poly'��Ч��?
��6��probablity:?�����Թ����Ƿ�ʹ��(true?or?false)��?
��7��shrinking���Ƿ��������ʽ��?
��8��tol��default?=?1e?-?3��:?svm������׼�ľ���;?
��9��cache_size:?�ƶ�ѵ������Ҫ���ڴ棨��MBΪ��λ����?
��10��class_weight:?ÿ������ռ�ݵ�Ȩ�أ���ͬ�������ò�ͬ�ĳͷ�����C,?ȱʡ�Ļ�����Ӧ��?
��11��verbose:?�����߳��йأ���������ɶ��˼���壻?
��12��max_iter:?������������default?=?1��?if?max_iter?=?-1,?no?limited;?
��13��decision_function_shape?��?��ovo��?һ��һ,?��ovr��?��Զ�??or?None?��,?default=None?
��14��random_state?�����ڸ��ʹ��Ƶ���������ʱ��α����������������ӡ�?

��ʾ��7,8,9һ�㲻���ǡ�?

def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
 
C,������ĳͷ�ϵ�������Էִ������ĳͷ��̶�Խ�������ѵ��������׼ȷ��Խ�ߣ����Ƿ����������ͣ�Ҳ���ǶԲ������ݵķ���׼ȷ�ʽ��͡�
kernel���㷨�в��õĺ˺�������
degree���������ֻ�Զ���ʽ�˺������ã���ָ����ʽ�˺����Ľ���n
gamma���˺���ϵ����Ĭ��Ϊauto
coef0���˺����еĶ�����
probability���Ƿ����ø��ʹ���
shrinking���Ƿ��������ʽ������ʽ
tol��svmֹͣѵ��������
cache_size��ָ��ѵ������Ҫ���ڴ棬��MBΪ��λ��Ĭ��Ϊ200MB��
class_weight����ÿ�����ֱ����ò�ͬ�ĳͷ�����C�����û�и�������������𶼸�C=1����ǰ�����ָ���Ĳ���C.
verbose���Ƿ�������ϸ���������������libsvm�е�ÿ����������ʱ���ã�������ã������޷��ڶ��߳�������������������һ���������ΪFalse�����ù�����
max_iter�����������������Ϊ-1����ʾ������
random_state��α�����������������,�ڻ�ϴ����ʱ���ڸ��ʹ��ơ�
���ԣ�
    svc.n_support_��������ж��ٸ�֧������
    svc.support_�������֧��������ѵ�������е�����
    svc.support_vectors_���������е�֧������
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

#�����б����
lda = LinearDiscriminantAnalysis()
'''
__init__����
    def __init__(self, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver

solver="svd",����㷨��svd��ʾʹ������ֵ�ֽ���⣬���ü���Э�������
             lsqr��ʾ��Сƽ��qr�ֽ�
             eigen��ʾ����ֵ�ֽ�
shrinkage=None,�Ƿ�ʹ�ò�������
priors=None,����LDA�б�Ҷ˹������������
components,��Ҫ����������������С�ڵ���n-1
store_covariance���Ƿ����ÿ�����Э�������0.19�汾ɾ��
�÷���
    lda.fit(X_train, y_train)
���ԣ�
    covariances_��ÿ�����Э������� shape = [n_features, n_features]
    means_�����ֵ��shape = [n_classes, n_features]
    priors_����һ�����������
    rotations_��LDA�����õ������ᣬshape [n_features, n_component]
    scalings_�������б�ÿ����˹�ֲ��ķ����
'''

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
'''
__init__����
def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8):

hidden_layer_sizes������=n_layers-2, Ĭ��(100��������i��Ԫ�ر�ʾ��i�����ز����Ԫ�ĸ�����
activation���������Ĭ��Ϊrelu
solver��Ĭ�� ��adam���������Ż�Ȩ�� 
alpha����ѡ�ģ�Ĭ��0.0001,��������� 
batch_size��Ĭ�ϡ�auto��,����Ż���minibatches�Ĵ�С
learning_rate��Ĭ�ϡ�constant��������Ȩ�ظ���
max_iter��Ĭ��200�������������� 
random_state����ѡ��Ĭ��None���������������״̬������
shuffle����ѡ��Ĭ��True,ֻ�е�solver=��sgd�����ߡ�adam��ʱʹ�ã��ж��Ƿ���ÿ�ε���ʱ������������ϴ��
tol����ѡ��Ĭ��1e-4���Ż������̶� 
learning_rate_int��Ĭ��0.001����ʼѧϰ�ʣ����Ƹ���Ȩ�صĲ�����ֻ�е�solver=��sgd�� ��adam��ʱʹ�á� 
power_t��ֻ��solver=��sgd��ʱʹ�ã�������չѧϰ�ʵ�ָ��.��learning_rate=��invscaling��������������Чѧϰ�ʡ� 
verbose���Ƿ񽫹��̴�ӡ��stdout
warm_start�������ó�True��ʹ��֮ǰ�Ľ��������Ϊ��ʼ��ϣ������ͷ�֮ǰ�Ľ�������� 

���ԣ�
    - classes_:ÿ����������ǩ 
    - loss_:��ʧ������������ĵ�ǰ��ʧֵ 
    - coefs_:�б��еĵ�i��Ԫ�ر�ʾi���Ȩ�ؾ��� 
    - intercepts_:�б��е�i��Ԫ�ش���i+1���ƫ������ 
    - n_iter_ ���������� 
    - n_layers_:���� 
    - n_outputs_:����ĸ��� 
    - out_activation_:�������������ơ�
�÷���
    - fit(X,y):��� 
    - get_params([deep]):��ȡ���� 
    - predict(X):ʹ��MLP����Ԥ�� 
    - predic_log_proba(X):���ض������ʹ��� 
    - predic_proba(X)�����ʹ��� 
    - score(X,y[,sample_weight]):���ظ����������ݺͱ�ǩ�ϵ�ƽ��׼ȷ�� 
    -set_params(**params):���ò�����
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
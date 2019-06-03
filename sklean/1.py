from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

dataset=datasets.make_classification(n_samples=1000,
                                     n_features=10,n_informative=2,n_redundant=2,n_repeated=0,n_classes=2)
kf=KFold(n_splits=2)
for train_index,test_index in kf.split(dataset):
    X_train,y_train=dataset[0][train_index],dataset[1][train_index]
    X_test,y_test=dataset[0][test_index],dataset[1][test_index]
clf=GaussianNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
acc=metrics.accuracy_score(y_test,pred)
print('acc:'+str(acc))
f1=metrics.f1_score(y_test,pred)
print('f1:'+str(f1))
auc=metrics.accuracy_score(y_test,pred)
print('auc:'+str(auc))

clf=SVC(C=1e-01,kernel='rbf',gamma=0.1)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
print("\nSVC")
acc=metrics.accuracy_score(y_test,pred)
print('acc:'+str(acc))
f1=metrics.f1_score(y_test,pred)
print('f1:'+str(f1))
auc=metrics.roc_auc_score(y_test,pred)
print('auc:'+str(auc))
clf=RandomForestClassifier(n_estimators=6)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
print("\nRandomForest")
acc=metrics.accuracy_score(y_test,pred)
print('acc:'+str(acc))
f1=metrics.f1_score(y_test,pred)
print('f1:'+str(f1))
auc=metrics.roc_auc_score(y_test,pred)
print('auc:'+str(auc))

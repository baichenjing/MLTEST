import pandas as pd
import numpy as np
from sklearn import tree
train=pd.read_csv('./input/train.csv')
test=pd.read_csv('./input/test.csv')

print("train shape "+str(train.shape))
print("train shape "+str(train.shape))
print("test shape "+str(test.shape))
print("train columns "+str(train.columns))
print("test columns "+str(test.columns))

train.head()
ages=train.loc[~train['Age'].isna()].append(test.loc[~test['Age'].isna()],sort=False)
ages.head()



def groupby_age_sex(row):
    if 'Mr.' in row['Name']:
        return 0
    elif 'Master.' in row['Name']:
        return 1
    elif 'Mrs.' in row['Name']:
        return 2
    elif 'Miss.' in row['Name']:
        return 3
    elif 'Ms.' in row['Name']:
        return 3
    elif 'Dr.' in row['Name']:
        if row['Sex']=='Male':
            return 0
        else:
            return 2

ages['group']=ages.apply(lambda x:groupby_age_sex(x),axis=1)
ages.head()

ages=ages.groupby('group').agg({'Age':'mean'})
ages.rename(columns={'Age':'Ave Age'},inplace=True)
ages.head()

test['group']=test.apply(lambda x:groupby_age_sex(x),axis=1)
train['group']=train.apply(lambda x:groupby_age_sex(x),axis=1)

test.set_index('group',inplace=True)
train.set_index('group',inplace=True)

test=test.join(ages,how='left')
train=train.join(ages,how='left')

test.loc[test['Age'.isna(),'Age']]=test.loc[test['Age'].isna()]['Age']
test.drop('Avg Age',axis=1,inplace=True)
test.head()

train.loc[train['Age'].isna(),'Age']=train.loc[train['Age'].isna()]['Avg Age']
train.drop('Avg Age',axis=1,inplace=True)
train.head()

train=train.fillna(0)
test=test.fillna(0)

train.reset_index(inplace=True)
train_target=train['Survived'].copy()
train_results=train[['PassengerId','Survived']].copy()
train.drop(['group','Name','Ticket','PassengerId','SibSp','Parch','Surviced'],axis=1,inplace=True)

train.head()

test.reset_index(inplace=True)
test_results=test[['PassengerId']].copy()
test.drop(['group','Name','Ticket','PassengerId','SibSp','Parch'],axis=1,inplace=True)
test.head()


def build_mapping(arr):
    arr=sorted(set(arr))
    i=0
    for ea in arr:
        i+=1
        try:
            item_map.update({ea:1})
        except:
            item_map={ea:1}

train['Sex']=train['Sex'].astype(str)
train['Embarked']=train['Embarked'].astype(str)
train['Cabin']=test['Cabin'].astype(str)

map_sex=build_mapping(train['Sex'])
map_embarked=build_mapping(train['Embarked'].astype(str))

cabin_list=train['Cabin'].append(test['Cabin'])
cabin_list.sort_values(inplace=True)

map_cabin=build_mapping(cabin_list)

train['Sex']=train['Sex'].map(map_sex)
train['Cabin']=train['Cabin'].map(map_cabin)
train['Embarked']=train['Embarked'].map(map_embarked)

dt=tree.DecisionTreeClassifier(min_samples_split=30)
dt=dt.fit(train,train_target)



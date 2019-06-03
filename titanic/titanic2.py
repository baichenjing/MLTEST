import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense,Dropout

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOGLEVEL']='2'
np.random.seed(10)

filepath='./input/train.csv'
all_df=pd.read_csv(filepath)
cols = ["Survived","Name","Pclass","Sex", "Age", "Parch", "Fare", "Embarked"]
all_df=all_df[cols]

def preprocess_data(raw_df):
    df=raw_df.drop(['Name'],axis=1)

    age_mean=df['Age'].mean()
    df['Age']=df['Age'].fillna(age_mean)
    fare_mean=df['Fare'].mean()
    df['Fare']=df['Fare'].mean()
    df['Sex']=df['Sex'].map({'female':0,'male':1}).astype(int)
    x_onehot_df=pd.get_dummies(data=df,columns=['Embarked'])

    ndarray=x_onehot_df.values
    label=ndarray[:,0]
    features=ndarray[:,1:]
    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled_feature=minmax_scale.fit_transform(features)

    return scaled_feature,label

msk=np.random.rand(len(all_df))<0.8
train_df=all_df[msk]
test_df=all_df[~msk]

train_features,train_label=preprocess_data(train_df)
test_features,test_label=preprocess_data(test_df)

print(train_features[:3])
print(test_features[:3])

model=Sequential()
model.add(Dense(
    units=40,
    input_dim=8,
    kernel_initializer="uniform",
    activation="relu"
))
model.add(
    Dense(
        units=30,
        kernel_initializer="uniform",
        activation="relu"
    )
)
model.add(Dense(
    units=1,
    kernel_initializer="uniform",
    activation="sigmoid"
) )
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)
train_history=model.fit(
    x=train_features,
    y=train_label,
    validation_split=0.1,
    epochs=30,
    batch_size=30,
    verbose=2
)

def show_train_history(train_history,train,val):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epochs")
    plt.legend(["train","validation"],loc="upper left")
    plt.show()

show_train_history(train_history,"acc","val_acc")
show_train_history(train_history,"loss","val_loss")

scores=model.evaluate(x=test_features,y=test_label)
print(" loss:",scores[0])
print("accuracy:",scores[1])

jack=pd.Series([0,'Jack',3,'male',23,0,5.0000,'S'])
rose=pd.Series([1,'Rose',1,'female',20,0,100.0000,'S'])

jr_df=pd.DataFrame([list(jack),list(rose)],columns=cols)
all_df=pd.concat([all_df,jr_df])
print(all_df[-2:])
all_features,all_label=preprocess_data(all_df)
all_probability=model.predict(all_features)
print(all_probability[:10])

print("*"*20)
new_df=all_df
new_df.insert(len(all_df.columns),"probability",all_probability)
print(new_df[-2:])

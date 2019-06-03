import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime,date
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

env=twosigmanews.make_env()
print('Done')

(market_train_df,news_train_df)=env.get_training_data()
market_train_df['time']=market_train_df['time'].dt.date
market_train_df=market_train_df.loc[market_train_df['time']>=date(2018,1,1)]

from multiprocessing import Pool

def create_lag(df_code,n_lag=[3,7,14],shift_size=1):
    code=df_code['assetCode'].unique()
    for col in return_features:
        for window in n_lag:
            rolled=df_code[col].shift(shift_size).rolling(window=window)
            lag_mean=rolled.mean()
            lag_max=rolled.max()
            lag_min=rolled.min()
            lag_std=rolled.std()
            df_code['%s_lag_%s_mean'%(col,window)]=lag_mean
            df_code['%s_lag_%s_max'%(col,window)]=lag_max
            df_code['%s_lag_%s_min'%(col,window)]=lag_min
        return df_code.fillna(-1)
def generate_lag_features(df,n_lag=[3,7,14]):
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                'returnsOpenNextMktres10', 'universe']
    assetCodes=df['assetCode'].unique()
    print(assetCodes)
    all_df=[]
    df_codes=df.groupby('assetCode')
    df_codes=[df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    print('total %s df'%len(df_codes))
    pool=Pool(4)
    all_df=pool.map(create_lag,df_codes)

    new_df=pd.concat(all_df)
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()

    return new_df

return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
n_lag=[3,7,14]
new_df=generate_lag_features(market_train_df,n_lag=n_lag)
market_train_df=pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
print(market_train_df.columns)

def mis_impute(data):
    for i in data.comumns:
        if data[i].dtype=="object":
            data[i]=data[i].fillna("other")
        elif(data[i].dtype=="int64" or data[i].dtype=="float64"):
            data[i]=data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df=mis_impute(market_train_df)

def data_prep(market_train):
    lbs={k:v for v,k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT']=market_train['assetCode'].map(lbl)
    market_train=market_train.dropna(axis=0)
    return market_train

market_train_df=data_prep(market_train_df)
print(market_train_df.shape)

from sklearn.preprocessing import LabelEncoder
up=market_train_df['returnsOpenNextMktres10']

universe=market_train_df['universe'].values
d=market_train_df['time']

fcol=[c for c in market_train_df if c not in  ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences',
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X=market_train_df[fcol].values
up=up.values
r=market_train_df.returnsOpenNextMktres10.values
mins=np.min(X,axis=0)
maxs=np.max(X,axis=0)
rng=maxs-mins
X=1-((maxs-X)/rng)

assert  X.shape[0]==up.shape[0]==r.shape[0]

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

te=market_train_df['time']>date(2015,1,1)
tt=0
for tt,i in enumerate(te.values):
    if i:
        idx=tt
        print(i,tt)
        break
print(idx)

X_train,X_test=X[:idx],X[idx:]
up_train,up_test=up[:idx],up[idx:]
r_train,r_test=r[:idx],r[idx:]
u_train,u_test=universe[:idx],universe[idx:]
d_train,d_test=d[:idx],d[idx:]

train_data=lgb.Dataset(X_train,label=up_train.astype(int))
test_data=lgb.Dataset(X_test,label=up_test.astype(int))

x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]
def exp_loss(p,y):
    y=y.get_label()
    grad=-y*(1.0-1.0/1.0+np.exp(-y*p))
    hess=-(np.exp(y*p)*(y*p-1)-1)/((np.exp(y*p)+1)**2)
    return grad,hess

params_1={
    'task':'train',
    'boosting_type':'gbdt',
    'objective':'binary',
    'leaning_rate':x_1[0],
    'num_leaves':x_1[1],
    'min_data_in_leaf':x_1[2],
    'num_iteration':239,
    'max_bin':x_1[4],
    'verbose':1
}

params_2={
    'task':'train',
    'boosting_type':'gbot',
    'objective':'binary',
    'leaning_rate':x_2[0],
    'num_leaves':x_2[1],
    'min_data_in_leaf':x_2[2],
    'num_iteration':172,
    'max_bin':x_2[4],
    'verbose':1
}

gbm_1=lgb.train(params_1,
                train_data,
                num_boost_round=100,
                valid_sets=test_data,
                early_stopping_rounds=5,
                fobj=exp_loss
)

gbm_2=lgb.train(params_2,train_data,num_boost_round=100,valid_sets=test_data,early_stopping_rounds=5,
                fobj=exp_loss)
confidence_test=(gbm_1.predict(X_test)+gbm_2.predict(X_test))/2
confidence_test=(confidence_test-confidence_test.min())/(confidence_test.max()-confidence_test.min())
confidence_test=confidence_test*2-1
print(max(confidence_test),min(confidence_test))

r_test=r_test.clip(-1,1)
x_t_i=confidence_test*r_test*u_test
data={'day':d_test,'x_t_i':x_t_i}
df=pd.DataFrame(data)
x_t=df.groupby('day').sum().values.flattern()
mean=np.mean(x_t)
std=np.std(x_t)
score_test=mean/std
print(score_test)

import gc
del X_train,X_test
gc.collect()
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary,DataParser
from matplotlib import pyplot as plt
import config
from AFM import AFM

def load_data():
    dfTrain=pd.read_csv(config.TRAIN_FILE)
    dfTest=pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols=[c for c in df.columns if c not in ['id','target']]
        df['missing_feat']=np.sum((df[cols]==-1).values,axis=1)


#one hot 编码没有距离
import pandas as pd

def load_data(download=True):
    if download:
        data_path,_=urlretrieve("http://archive.ics.uci.edu")
        print('download to car.csv')
    col_names=['buying']
    data=pd.read_csv("car.csv",names=col_names)
    return data

def convert2onehot(data):
    return pd.get_dummies()


if __name__=="__main__":
    data=load_data(download=False)
    new_data=convert2onehot(data)

    print(data.head())
    print('num of data',len(data),'\n')

    for name in data.keys():
        print(name.pd.unique(data[name]))
    print('\n',new_data.head(2))
    new_data.to_csv("car_onehot.csv",index=False)
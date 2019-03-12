import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from pandas import np

if __name__=="__main__":
    show=True
    path='./Advertising.csv'
    data=pd.read_csv(path)
    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']
    print('Persone Corr=\n',data.corr())
    print(x)
    print(y)
    print(x.shape,y.shape)
    mpl.rcParams['font.sans-serif']=['simHei']
    mpl.rcParams['axes.unicode_minus']=False
    if show:
        #绘制1
        plt.figure(facecolor='white')
        plt.plot(data['TV'],y,'ro',label='TV',mec='k')
        plt.plot(data['Radio'],y,'g^',mec='k',label='Radio')
        plt.plot(data['Newspaper'],y,'mv',mec='k',label='Newspaper')
        plt.legend(loc='lower right')
        plt.xlabel('广告花费',fontsize=16)
        plt.ylabel('销售额',fontsize=16)
        plt.title('广告花费与销售额对比数据',fontsize=18)
        plt.grid(b=True,ls=':')
        plt.show()

        #绘制2
        plt.figure(facecolor='w',figsize=(9,10))
        plt.subplot(311)
        plt.plot(data['TV'],y,'ro',mec='k')
        plt.title('TV')
        plt.grid(b=True,ls=':')
        plt.subplot(312)
        plt.plot(data['Radio'],y,'g^',mec='k')
        plt.title('Radio')
        plt.grid(b=True,ls=':')
        plt.subplot(313)
        plt.plot(data['Newspaper'],y,'b*',mec='k')
        plt.title('Newspaper')
        plt.grid(b=True,ls=':')
        plt.tight_layout(pad=2)
        plt.show()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
    model=LinearRegression()
    model.fit(x_train,y_train)
    print(model.coef_,model.intercept_)

    order=y_test.argsort(axis=0)
    y_test=y_test.values[order]
    x_test=x_test.values[order,:]
    y_test_pred=model.predict(x_test)
    mse= pd.np.mean((y_test_pred - pd.np.array(y_test)) ** 2)
    rmse= pd.np.sqrt(mse)
    mse_sys=mean_squared_error(y_test,y_test_pred)
    print('MSE= ',mse,end=' ')
    print('MSE(System Function) = ',mse_sys,end=' ')
    print(mean_absolute_error(y_test,y_test_pred))

    error = y_test - y_test_pred
    np.set_printoptions(suppress=True)
    print('error = ', error)
    plt.hist(error, bins=20, color='g', alpha=0.6, edgecolor='k')
    plt.show()

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_test_pred, 'g-', linewidth=2, label='预测数据')
    plt.legend(loc='upper left')
    plt.title('线性回归预测销量', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()










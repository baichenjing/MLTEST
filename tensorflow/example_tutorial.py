import numpy as np
import matplotlib as plt
import tensorflow as tf
LR=0.1
REAL_PARAMS=[1.2,2.5]
INIT_PARAMS=[[5,4],[5,1],[2,4.5]][2]
x=np.linespace(-1,1,200,dtype=np.float32)

y_fun=lambda a,b:a*x+b
tf_y_fun=lambda a,b:a*x+b

noise=np.random.randn(200)/10
y=y_fun(*REAL_PARAMS)+noise

a,b=[tf.Variable(initial_value=p,dtype=tf.float32) for p in INIT_PARAMS]
plt.scatter(x,y)
plt.show()


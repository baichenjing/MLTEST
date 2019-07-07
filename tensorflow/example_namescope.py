import tensorflow as tf
from __future__ import print_function
tf.set_random_seed(1)

with tf.name_scope('a_name_scope'):
    initializer=tf.constant_initializer(value=1)
    var1=tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=initializer)
    var2=tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
    var21=tf.Variable(name='var2',initial_value=[2,1],dtype=tf.float32)
    var22=tf.Variable(name='var2',initial_value=[2,2],dtype=tf.float32)

with tf.variable_scope('a_name_scope'):
    initializer=tf.constant_initializer(value=1)
    var1=tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=initializer)
    var2=tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
    var21=tf.Variable(name='var2',initial_value=[2,1],dtype=tf.float32)
    var22=tf.Variable(name='var2',initial_value=[2,2],dtype=tf.float32)

with tf.variable_scope('a_name_scope'):
    tf.scope.reuser_varibales()

with tf.Session() as sess:
    print(var1.name)
    print(sess.run(var1))
    print(var2.name)



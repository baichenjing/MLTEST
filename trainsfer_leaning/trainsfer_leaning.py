import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

def text2vec(labels):
    number=['0','1','2','3','4','5','6','7','8','9']
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']

    dictionary=number+alphabet
    label_new=[]
    for label in labels:
        y_name=list(label)
        y_label=[]
        for i in range(6):
            y_label.append(np.zeros(len(dictionary)))
        y_label=np.array(y_label)

        for i in range(len(y_name)):
            if y_name[i]=='.':
                continue

            key=dictionary.index(y_name[i])
            y_label[i][key]=1
        y_label=y_label.reshape(-1)
        label_new.append(y_label)
    return label_new

def vec2text(y_vectors):
    number= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    dictionary=number+alphabet
    y_names=[]

    for y_vector in y_vectors:
        y_name_new=[]
        for y_key in y_vectors:
            y_name_new=[]
            for y_key in y_vector:
                y_name_new.append(dictionary[int(y_key)])
            y_name=''.join(y_name_new)
            y_names.append(y_name)
        return y_names

def load_data(img_dir):
    data=[]
    labels=[]
    img_name=os.listdir(img_dir)

    for i in range(len(img_name)):
        path=os.path.join(img_dir,img_name[i])
        image=cv2.imread(path)
        data.append(image)
        y_temp=img_name[i][:6]
        labels.append(y_temp)

    label_new=text2vec(labels)
    x=np.array(data)
    y=np.array(label_new)

    return x,y

def load_predict_data(img_dir):
    data=[]
    img_name=os.listdir(img_dir)
    for i in range(len(img_name)):
        path=os.path.join(img_dir,img_name[i])
        image=cv2.imread(path)
        data.append(image)
    x=np.array(data)
    return x

def load_test_data(img_dir):
    data=[]
    labels=[]
    img_name=os.listdir(img_dir)

    for i in range(len(img_name)):
        path=os.path.join(img_dir,img_name[i])
        image=cv2.imread(path)
        data.append(image)
        y_temp=img_name[i][:6]
        labels.append(y_temp)
    x=np.array(data)
    y=np.array(labels)
    return x,y

class Vgg16:
    vgg_mean=[103.939,116.779,123.68]

    def __init__(self,vgg16_npy_path=None,restore_from=None):
        try:
            self.data_dict=np.load(vgg16_npy_path,encoding='latin1').item()
        except FileNotFoundError:
            print('please download VGG16 parameters from here')

        self.tfx=tf.placeholder(tf.float32,[None,224,224,3])
        self.tfx=tf.placeholder(tf.float32,[None,216])
        self.keep_prob=tf.placeholder(tf.float32)

        red,green,blue=tf.split(axis=3,num_or_size_splits=3,value=self.tfx*255.0)
        bgr=tf.concat(axis=3,values=[blue-self.vgg_mean[0],
                                     green-self.vgg_mean[1],
                                     red-self.vgg_mean[2],])

        conv1_1=self.conv_layer(bgr,'conv1_1')
        conv1_2=self.conv_layer(conv1_1,"conv1_2")
        pool1=self.max_pool(conv1_2,'pool1')

        conv1_1=self.conv_layer(bgr,'conv1_1')
        conv1_2=self.conv_layer(conv1_1,"conv1_2")
        pool1=self.max_pool(conv1_2,'pool1')

        conv2_1=self.conv_layer(pool1,'conv2_1')
        conv2_2=self.conv_layer(conv2_1,"conv2_2")
        pool2=self.max_pool(conv2_2,'pool2')

        conv3_1=self.conv_layer(pool2,'conv3_1')
        conv3_2=self.conv_layer(conv3_1,'conv3_2')
        conv3_3=self.conv_layer(conv3_2,"conv3_3")
        pool3=self.max_pool(conv3_3,'pool3')

        conv4_1=self.conv_layer(pool3,"conv4_1")
        conv4_2=self.conv_layer(conv4_1,"conv4_2")
        conv4_3=self.conv_layer(conv4_2,"conv4_3")
        pool4=self.max_pool(conv4_3,'pool4')

        conv5_1=self.conv_layer(pool4,'conv5_1')
        conv5_2=self.conv_layer(conv5_1,"conv5_2")
        conv5_3=self.conv_layer(conv5_2,"conv5_3")
        pool5=self.max_pool(conv5_3,'pool5')

        with tf.variable_scope('new_train'):
            self.flatten=tf.reshape(pool5,[-1,7*7*512])
            self.fc6=tf.layers.dense(self.flatten,512)
            self.out=tf.layers.dense(self.fc6,216,name='out')

        self.y_pre=tf.reshape(self.out,[-1,6,36])
        self.y_predict_vec=tf.argmax(self.y_pre,2)
        y_label=tf.reshape(self.tfy,[-1,6,36])
        correct_pred=tf.equal(tf.argmax(self.y_pre,2),tf.argmax(y_label,2))
        self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        config=tf.ConfigProto(device_count={"CPU":2},
                              inter_op_parallelism_threads=2,
                              intra_op_parallelism_threads=2,
                              log_device_placement=True)
        self.sess=tf.Session(config=config)
        if restore_from:
            saver=tf.train.Saver()
            saver.restore(self.sess,restore_from)
        else:
            diff=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out,labels=self.tfy)
            self.loss=tf.reduce_mean(diff)

            output_layer_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES,scope="new_train")
            optimizer=tf.train.AdamOptimizer(learning_rate=0.0001,name="Adam2")
            self.train_op=optimizer.minimize(self.loss,var_list=output_layer_vars)
            self.sess.run(tf.global_varibales_initializer())

        def max_pool(self,bottom,name):
            return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SNAME',name=name)

        def conv_layer(self,bottom,name):
            with tf.varibale_scope(name):
                conv=tf.nn.conv2d(bottom,self.data_dict[name][0],[1,1,1,1],padding='SNAME')
                lout=tf.nn.relu(tf.nn.bias_add(conv,self.data_dict[name][1]))
                return lout

        def train(self,x,y,keep_prob):
            loss,_=self.sess.run([self.loss,self.train_op],{self.tfx:x,self.tfy:y,self.keep_prob:keep_prob})
            return loss

        def predict(self,img_dir):
            x,y=load_test_data(img_dir)
            y_predict=self.sess.run([self.y_predict_vec],{self.tfx:x})
            return y_predict,y

        def predict_sigle(self,img_dir):
            x=load_predict_data(img_dir)
            y_predict=self.sess.run([self.y_predict_vec],{self.tfx:x})
            return y_predict

        def save(self,path='',step=1):
            saver=tf.train.Saver()
            saver.save(self.sess,path,write_meta_graph=False,global_step=step)

def train():
    img_dir=''
    x,y=load_data(img_dir)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.016)
    vgg=Vgg16(vgg16_npy_path='')
    for i in range(20000):
        b_idx=np.random.randint(0,len(x_train),100)
        if (i+1)%100==0:
            train_accuracy=vgg.sess.run([vgg.accuracy],{vgg.tfx:x_test[:100],vgg.tfy:y_test[:100],vgg.keep_prob:1.0})
        train_loss=vgg.train(x_train[b_idx],y_train[b_idx],keep_prob=0.5)
        if (i+1)%200==0:
            test_accuracy=vgg.sess.run([vgg.accuracy],{vgg.tfx:x_test[:100],vgg.tfy:y_test[:100],vgg.keep_prob:1.0})
            test_accuracy=vgg.sess.run([vgg.accuracy],{vgg.tfx:x_test[100:200],vgg.tfy:y_test[100:200],vgg.keep_prob:1.0})

        if (i+1)%200==0 and test_accuracy[0]>0.7:
            vgg.save('./for_transfer_leaning/model/transfer_lean',i)

        def evaluate():
            vgg=Vgg16(vgg16_npy_path='vgg16.npy',
                      restore_from='./for_transfer_leaning/models/transfer_lean-799')
            img_dir='./img_down_sets/img_test_border'
            y_predict,y=vgg.predict(img_dir)
            y_predict=np.array(y_predict)

            y_name=vec2text(y_predict[0])
            count=0
            for i in range(len(y_name)):
                if y[i]==y_name[i]:
                    count+=1
            accuracy=(count/len(y_name))*100

        def predict():
            vgg=Vgg16(vgg16_npy_path='',
                      restore_from='./for')
            img_dir='./img_down_sets'
            y_predict=vgg.predict_sigle(img_dir)
            y_name=vec2text(y_predict[0])
            for i in range(len(y_name)):
                print()

            return y_name[0]

    if __name__=='__main__':
        train()

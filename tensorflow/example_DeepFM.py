import tensorflow as tf
class DeepFM(object):
    def __init__(self):
        self.feat_index=tf.placeholder(tf.int32,
                                       shape=[None,None],
                                       name='feat_index')
        self.feat_value=tf.placeholder(tf.float32,
                                       shape=[None,None],
                                       name='feat_value')
        self.label=tf.placeholder(tf.float32,shape=[None,1],name='label')
        self.dropout_keep_fm=tf.placeholder(tf.float32,shape=[None],name='dropout_keep_fm')
        self.dropout_keep_deep=tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')

        self.weights['feature_embeddings']=tf.Variable(tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')
        self.weights['feature_bias']=tf.Variable(tf.random_normal([self.feature_size,1],0.0,1.0),name='feature_bias')
        self.embeddings=tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index)
        feat_value=tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
        self.embeddings=tf.multiply(self.embeddings,feat_value)

        self.y_first_order=tf.nn.embedding_lookup(self.weights['feature_bias'],self.feat_index)
        self.y_first_order=tf.reduce_sum(tf.multiply(self.y_first_order,feat_value),2)
        self.y_first_order=tf.nn.dropout(self.y_first_order,self.dropout_keep_fm[0])

        self.summed_features_emb=tf.reduce_sum(self.embeddings,1)
        self.summed_features_emb_square=tf.square(self.summed_reatures_emb)

        self.squared_featured_emb=tf.square(self.embeddings)
        self.squared_sum_features_emb=tf.reduce_sum(self.squared_featured_emb,1)

        self.y_second_order=0.5*tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb)

        self.y_second_order=0.5*tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb)
        self.y_second_order=tf.nn.dropout(self.y_second_order,self.dropout_keep_fm[1])

        self.y_deep=tf.reshape(self.embeddings,shape=[-1,self.field_size*self.embedding_size])
        self.y_deep=tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])
        for i in range(0,len(self.deep_layers)):
            self.y_deep=tf.add(tf.matmul(self.y_deep,self.weights['layer_%d'%i]))
            self.y_deep=self.deep_layers_activation(self.y_deep)
            self.y_deep=tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])
        concat_input=tf.concat([self.y_first_order,self.y_second_order,self.y_deep],axis=1)

        if self.loss_type=='logloss':
            self.out=tf.nn.sigmoid(self.out)
            self.loss=tf.losses.log_loss(self.label,self.out)
        elif self.loss_type=="mse":
            self.loss=tf.nn.l2_loss(tf.subtract(self.lable,self.out))

        if self.l2_reg>0:
            self.loss+=tf.contrib.layers.l2_regularizer(
                self.l2_reg)(self.weights['concat_projection'])
            if self.use_deep:
                for i in range(len(self.deep_layers)):
                    self.loss+=tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layers_%d"%i])

        if self.optimizer_type=="adam":
            self.optimizer=tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,beta1=0.9,
                beta2=0.999,epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type=="adagrad":
            self.optimizer=tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                     initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type=='gd':
            self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
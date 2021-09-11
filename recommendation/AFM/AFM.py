import tensorflow as tf
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import roc_auc_score

class AFM(BaseEstimator,TransformerMixin):
    def __init__(self,feature_size,field_size,
                 embedding_size=8,attention_size=10,
                 deep_layers=[32,32],deep_init_size=50,
                 dropout_deep=[0.5,0.5,0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10,batch_size=256,
                 learning_rate=0.001,optimizer="adam",
                 batch_norm=0,batch_norm_decay=0.995,
                 verbose=False,random_seed=2016,
                 loss_type="logloss",eval_metric=roc_auc_score,greater_is_better=True,
                 use_inner=True):
        assert loss_type in ["logloss","mse"]

        self.feature_size=feature_size
        self.field_size=field_size
        self.embedding_size=embedding_size
        self.attention_size=attention_size

        self.deep_layer=deep_layers
        self.deep_init_size=deep_init_size
        self.dropout_dep=dropout_deep
        self.deep_layers_activation=deep_layer_activation

        self.epoch=epoch
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.optimizer_type=optimizer

        self.batch_norm=batch_norm
        self.batch_norm_decay=batch_norm_decay

        self.verbose=verbose
        self.randrom_seed=random_seed
        self.loss_type=loss_type
        self.eval_metric=eval_metric
        self.greater_is_better=greater_is_better
        self.train_result,self.valid_result=[],[]

        self.user_inner=use_inner
        self._init_graph()

    def _init_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index=tf.placeholder(tf.int32,
                                           shape=[None,None],
                                           name='feat_index')
            self.feat_value=tf.placeholder(tf.float32,
                                           shape=[None,None],
                                           name='feat_value')
            self.labe=tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_deep=tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phrase=tf.placeholder(tf.bool,name='train_phase')

            self.weights=self._initilize_weights()

            self.embeddings=tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index)
            feat_value=tf.reshape(self.feate_value,shape=[-1,self.field_size,1])




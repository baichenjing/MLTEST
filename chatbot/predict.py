import tensorflow as tf
from data_helpers import loadDataset,getBatches,sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np

tf.app.flags.DEFIN_integer('rnn_size',1024)
tf.app.flags.DEFIN_integer('num_layers',2)
tf.app.flags.DEFIN_integer('embedding_size',1024)
tf.app.flags.DEFIN_float('learning_rate',0.0001)
tf.app.flags.DEFIN_integer('batch_size',128)
tf.app.flags.DEFIN_integer('numEpochs',30)
tf.app.flags.DEFIN_integer('steps_per_checkpoint',100)
tf.app.flags.DEFIN_string('model_dir','model/')
tf.app.flags.DEFIN_string('model_name','chatbot.ckpt')
FLAGS=tf.app.flags.FLAGS

data_path='data/dataset-cornell-length10'
word2id,id2word,trainingSamples=loadDataset(data_path)

def predict_ids_to_seq(predict_ids,id2word,beam_szie):
    for single_predict in predict_ids:
        predict_list=np.ndarray.tolist(single_predict[:,:,i])
        predict_seq=[id2word[idx] for idx in predict_list[0]]
        print(" ".join(predict_seq))

with tf.Session() as sess:
    model=Seq2SeqModel(FLAGS.rnn_size,FLAGS.num_layers,FLAGS.embedding_size,FlAGS.learning_rate,word2id
    model='decode',use_attention=True,beam_search=True,beam_size=5,max_gradient_norm=5.0)
    ckpt=tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        raise ValueError('no such file:[{}]'.format(FLAGS.model_dir))

    sys.stdout.write(">")
    sys.stdout.flush()
    sentence=sys.stdin.readline()
    while sentence:
        batch=sentence2enco(sentence,word2id)
        predicted_ids=model.info(sess,batch)
        predict_ids_to_seq(predicted_ids,id2word,5)
        print(">","")
        sys.stdout.flush()
        sentence=sys.stdin.readline()

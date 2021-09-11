import tensorflow as tf
from data_loader import loadDataset,getBatches,sentence2enco
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os

tf.app.flags.DEFINE_integer('rnn_size',1024)
tf.app.flags.DEFINE_integer('num_layers',2)
tf.app.flags.DEFINE_integer('embedding_size',1024)
tf.app.flags.DEFINE_float('learning_rate',0.0001)
tf.app.flags.DEFINE_integer('batch_size',128)
tf.app.flags.DEFINE_integer('numEpochs',30)
tf.app.flags.DEFINE_integer('steps_per_checkpoint',100)
tf.app.flags.DEFINE_string('model_dir','model/')
tf.app.flags.DEFINE_string('model_name','chatbot.ckpt')

FLAGS=tf.app.flags.FLAGS

data_path=''
word2id,id2word,trainingSamples=loadDataset(data_path)
with tf.Session() as sess:
    model=Seq2SeqModel(
        FLAGS.rnn_size,
        FLAGS.num_layers,
        FLAGS.embedding_size,
        FLAGS.learning_rate,
        word2id,
        mode='train',
        use_attention=True,
        beam_search=True,
        beam_size=5,
        max_gradient_norm=5.0
    )
    ckpt=tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    current_step=0
    summary_writer==tf.summary.FileWriter(FLAGS.model_dir,graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        batches=getBatches(trainigSamples,FLAGS.batch_size)
        for nextBatch in tqdm(batches,desc='Training'):
            loss,summary=model.train(sess.nextBatch)
            current_step+=1
            if current_step % FLAGS.steps_per_checkpoint==0:
                perplexity=math.exp(float(loss)) if loss<300 else float('inf')
                summary_writer.add_summary(summary,current_step)
                checkpoint_path=os.path.join(FLAGS.model_dir,FLAGS.model_name)
                model.saver.save(sess,checkpoint_path,global_step=current_step)


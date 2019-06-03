import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

parser=argparse.ArgumentParser()
parser.add_argument('--train_dir',default='train')
parser.add_argument('--val_dir',default='val')
parser.add_argument('--model_path',default='vgg_16.ckpt',type=str)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--num_workers',default=4,type=int)
parser.add_argument('--num_epochs1',default=10,type=int)
parser.add_argument('--num_epochs2',default=10,type=int)
parser.add_argument('--learning_rate1',default=1e-3,type=float)
parser.add_argument('--learning_rate2',default=1e-5,type=float)
parser.add_argument('--dropout_keep_prob',default=0.5,type=float)
parser.add_argument('--weight_decay',default=5e-4,type=float)

VGG_NAME=[123.68,116.78,103.94]

def list_images(directory):
    labels=os.listdir(directory)
    files_and_labels=[]
    for label in labels:
        for f in os.listdir(os.path.join(directory,label)):
            files_and_labels.append((os.path.join(directory,label,f),label))

    filenames,labels=zip(*files_and_labels)
    fielnames=list(filenames)
    labels=list(labels)
    unique_labels=list(set(labels))

    label_to_int={}
    for i,label in enumerate(unique_labels):
        label_to_int[label]=i

    labels=[label_to_int[l] for l in labels]

    return fielnames,labels

def check_accuracy(sess,correct_prediction,is_training,dataset_init_op):
    sess.run(dataset_init_op)
    num_correct,num_samples=0,0
    while True:
        try:
            correct_pred=sess.run(correct_prediction,{is_training:False})
            num_correct+=correct_pred.sum()
            num_samples+=correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    acc=float(num_correct)/num_samples
    return acc

def main(args):
    train_filenames,train_labels=list_images(args.train_dir)
    val_filenames,val_labels=list_images(args.val_dir)

    num_classes=len(set(train_labels))

    graph=tf.Graph()

    with graph.as_default():
        def _parse_function(filename,label):
            image_string=tf.read_file(filename)
            image_decoded=tf.image.decode_jpeg(image_string,channels=3)
            image=tf.cast(image_decoded,tf.float32)

            smallest_side=256.0
            height,width=tf.shape(image)[0],tf.shape(image)[1]
            height=tf.to_float(height)
            width=tf.to_float(width)

            scale=tf.cond(tf.greater(height,width),
                          lambda:smallest_side/width,
                          lambda:smallest_side/height)
            new_height=tf.to_int32(height*scale)
            new_width=tf.to_int32(width*scale)
            resized_image=tf.image.resize_images(image,[new_height,new_width])
            return resized_image,label

        def training_preprocess(image,label):
            crop_image=tf.random_crop(image,[224,224,3])
            flip_image=tf.image.random_flip_left_right(crop_image)

            means=tf.reshape(tf.constant(VGG_MEAN),[1,1,3])
            centered_image=flip_image-means

            return centered_image,label

        def val_preprocess(image,label):
            crop_image=tf.image.resize_image_with_crop_or_pad(image,224,224)
            means=tf.reshape(tf.constant(VGG_MEAN),[1,1,3])
            centered_image=crop_image-means

            return centered_image,label

        train_filenames=tf.constant(train_filenames)
        train_labels=tf.constant(train_labels)
        train_dataset=tf.contrib.data.Dataset.from_tensor_slices((train_filenames,train_labels))
        train_dataset=train_dataset.map(_parse_function,
                                        num_threads=args.num_workers,output_buffer_size=args.batch_size)
        train_dataset=train_dataset.shuffle(buffer_size=1000)
        batched_train_dataset=train_dataset.batch(args.batch_size)

        val_filenames=tf.constant(val_filenames)
        val_labels=tf.constant(val_labels)
        val_dataset=tf.contrib.data.Dataset.from_tensor_slices((val_filenames,val_labels))
        val_dataset=val_dataset.map(_parse_function,num_threads=args.num_workers,output_buffer_size=args.bathc_size)
        val_dataset=val_dataset.map(val_preprocess,num_threads=args.num_workers,output_buffer_size=args.batch_size)
        batched_val_dataset=val_dataset.batch(args.batch_size)

        iterator=tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                         batched_train_dataset.output_shapes)

        images,labels=iterator.get_next()

        train_init_op=iterator.make_initializer(batched_train_dataset)
        val_init_op=iterator.make_initializer(batched_val_dataset)

        is_training=tf.placeholder(tf.bool)
        vgg=tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits,_=vgg.vgg_16(images,num_classes=num_classes,is_training=is_training,
                                dropout_keep_prob=args.dropout_keep_prob)
        model_path=args.model_path
        assert(os.path.isfile(model_path))

        variables_to_restore=tf.contrib.framework.get_varibales_to_restore(exclude=['vgg_16/fc8'])
        init_fn=tf.contrib.framework.assign_from_checkpoint_fn(model_path,variables_to_restore)

        fc8_variables=tf.contrib.framework.et_varibales('vgg_16/fc8')
        fc8_init=tf.varibles_initializer(fc8_variables)

        tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        loss=tf.losses.get_total_loss()

        fc8_optimizer=tf.train.GradientDescentOptimizer(args.learning_rate1)
        fc8_train_op=fc8_optimizer.minimize(loss,var_list=fc8_variables)

        full_optimizer=tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op=full_optimizer.minimize(loss)

        prediction=tf.to_int32(tf.argmax(logits,1))
        correct_prediction=tf.equal(prediction,labels)
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        tf.get_default_graph().finalize()

        with tf.Session(gragp=graph) as sess:
            init_fn(sess)
            sess.fun(fc8_init)

            for epoch in range(args.num_epochs1):
                sess.run(train_init_op)
                while True:
                    try:
                        _=sess.run(fc8_train_op,{is_training:True})
                    except tf.errors.OutOfRangeError:
                        break

                train_acc=check_accuracy(sess,correct_prediction,is_training,train_init_op)
                val_acc=check_accuracy(sess,correct_prediction,is_training,val_init_op)


            for epoch in range(args.num_epch2):
                print('starting epoch %d/%d'%(epoch+1,args.num_epch2))
                sess.run(train_init_op)
                while True:
                    try:
                        _=sess.run(full_train_op,{is_training:True})
                    except tf.errors.OutOfRangeError:
                        break
                train_acc=check_accuracy(sess,correct_prediction,is_training,train_init_op)
                val_acc=check_accuracy(sess,correct_prediction,is_training,val_init_op)
                print('Training accuracy: %f'% train_acc)
                print('Val accuracy: %f\n'% val_acc)

if __name__=='__main__':
    args=parser.parse_args()
    main(args)
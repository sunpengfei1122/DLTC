#!/usr/bin/env python
# File: cifar-convnet.py
# Author: Yuxin Wu
import argparse
import os
from tensorpack import tfv1 as tf
import numpy as np
from tensorpack import *
from tensorpack.dataflow import dataset, BatchData
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu
import sys
import numpy as np
sys.path.append("..")
import SCNN1 
import os

from tensorpack.dataflow import dataset
from keras.utils import to_categorical

K = 100
K2 = 1e-2
TRAINING_BATCH = 10
scale = 2
class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def inputs(self):
        return [tf.TensorSpec((None, 28,28), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        image = scale*(-image + 1)
        print('input shape', image.shape)
        image = tf.reshape(tf.exp(image), [TRAINING_BATCH, 784])

        layer_in = SCNN1.SNNLayer(in_size=784, out_size=1000, n_layer=1, name='layer1')
        layer_out = SCNN1.SNNLayer(in_size=1000, out_size=10, n_layer=2, name='layer2')
       
        layerin_out = layer_in.forward(image)
        layerout_out = layer_out.forward(layerin_out)

        output_real = tf.one_hot(label, 10,dtype=tf.float32)
        layerout_groundtruth = tf.concat([layerout_out,output_real],1)
        loss = tf.reduce_mean(tf.map_fn(SCNN1.loss_func,layerout_groundtruth), name='cost')

        wsc = layer_in.w_sum_cost() + layer_out.w_sum_cost()
        l2c = layer_in.l2_cost() + layer_out.l2_cost()


        K = 100
        K2 = 1e-2
        cost = loss + K*wsc + K2*l2c
        tf.summary.scalar('cost', cost)
        correct = tf.cast(tf.nn.in_top_k(predictions=-layerout_out, targets=label, k=1), tf.float32, name='correct')
        

        
        # monitor training error
        add_moving_summary(tf.reduce_mean((correct), name='accuracy'))

        return cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-2,
            global_step=get_global_step_var(),
            decay_steps=int(50000/TRAINING_BATCH) ,
            decay_rate= (1e-4 / 1e-2) ** (1. / 70), staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)

def get_data(train_or_test, cifar_classnum=10, BATCH_SIZE=128):
    isTrain = train_or_test == 'train'  

    ds  = dataset.FashionMnist(train_or_test)      
    if isTrain:

        augmentors = [
	    imgaug.CenterPaste((32, 32)),
            imgaug.RandomCrop((28, 28)),
        ]
    else:
        augmentors = [
            #imgaug.CenterCrop((28, 28)),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config(cifar_classnum, BATCH_SIZE):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum,BATCH_SIZE)
    dataset_test = get_data('test', cifar_classnum, BATCH_SIZE)

    nr_tower = max(get_num_gpu(), 1)
    batch = BATCH_SIZE
    total_batch = batch * nr_tower
    print('total batch', total_batch)

 
    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)

    return  TrainConfig(        
         model=Model(cifar_classnum),
      
         data=input, 
         callbacks=[
             ModelSaver(),   # save the model after every epoch
             GPUUtilizationTracker(),
             EstimatedTimeLeft(),
             InferenceRunner(    # run inference(for validation) after every epoch
                 dataset_test,   # the DataFlow instance used for validation
                [ScalarStats('cost'), ClassificationError('correct')]),
                MaxSaver('validation__correct'),
		],
         #],
         max_epoch=70,
     )

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    #parser.add_argument('--load', help='load model')
    parser.add_argument('--classnum', help='10 for fmnist',
                        type=int, default=10)
    parser.add_argument('--batch', type=int, default=TRAINING_BATCH, help='batch per GPU')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        logger.set_logger_dir(os.path.join('train_log', 'delay' + str(args.classnum)))
        config = get_config(args.classnum, args.batch)

        num_gpu = get_num_gpu()
        print('total gpus', num_gpu)
        trainer = SimpleTrainer() if num_gpu <= 1 \
            else SyncMultiGPUTrainerParameterServer(num_gpu)
        launch_train_with_config(config, trainer)

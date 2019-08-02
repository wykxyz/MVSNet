#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
from random import randint
import visdom
import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
vis=visdom.Visdom(env="MVSNet")
sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import *
from homography_warping import get_homographies, homography_warping

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# paths
tf.app.flags.DEFINE_string('dtu_data_root', '/home/haibao637/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('log_dir', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time = time.strftime("%y-%m-%d")
tf.app.flags.DEFINE_string('logdir', os.path.join('/home/haibao637/data/tf_log/', time.strftime("%y-%m-%d"),
                                                   time.strftime("%H:%M:%S")),
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/haibao637/data/tf_models/'),
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step',50000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 640,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512,
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.06,
                          """Downsample scale for building cost volume.""")

# network architectures
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                            """Whether to apply depth map refinement for 1DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 10,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 1e-3,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 1250,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                           """Whether to apply inverse depth for R-MVSNet""")
FLAGS = tf.app.flags.FLAGS


class MVSGenerator:
    """ data generator class, tf only accept generator without param """

    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        random.shuffle(self.sample_list)
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0

    def __iter__(self):
        while True:
            for data in self.sample_list:
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                depth_images=[]
                # img = cv2.imread(data.image[0])
                for view in range(self.view_num):
                    images.append(center_image(cv2.imread(data.images[view])))
                    cams.append(load_cam(open(data.cams[view])))
                    depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                    depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                    depth_images.append(mask_depth_image(load_pfm(open(data.depths[view])),depth_start,depth_end))
                # for view in range(self.view_num):
                #
                #     image = center_image(cv2.imread(data[2 * view]))
                #     cam = load_cam(open(data[2 * view + 1]))
                #     cam[1][3][1] = cam[1][3][1] * (FLAGS.interval_scale)
                #     images.append(image)
                #     cams.append(cam)
                #     depth_image = load_pfm(open(data[2 * self.view_num]))
                #     depth_images.append(depth_image)

                # mask out-of-range depth pixels (in a relaxed range)
                # depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                # depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                # depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                depth_images=np.stack(depth_images,0)
                # depth_images=depth_images[...,np.newaxis]
                # print(depth_images.shape)
                #print('Forward pass: d_min = %f, d_max = %f.' % \
                #    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_images)

                # return backward mvs input for GRU
                # if FLAGS.regularization == 'GRU':
                #     self.counter += 1
                # start_time = time.time()
                # cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                # cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                # duration = time.time() - start_time
                # #print('Back pass: d_min = %f, d_max = %f.' % \
                # #    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                # yield (images, cams, depth_image)
        # while True:
        #     for data in self.sample_list:
        #         start_time = time.time()
        #         ###### read input data ######
        #         images = []
        #         cams = []
        #         for view in range(self.view_num):
        #             image = center_image(cv2.imread(data[2 * view]))
        #             cam = load_cam(open(data[2 * view + 1]))
        #             cam[1][3][1] = cam[1][3][1] * (FLAGS.interval_scale)
        #             if cam[1][3][2]==0:
        #                 cam[1][3][2]=FLAGS.max_d
        #             images.append(image)
        #             cams.append(cam)
        #         depth_image = load_pfm(open(data[2 * self.view_num]))
        #         # mask out-of-range depth pixels (in a relaxed range)
        #         depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
        #         depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
        #         depth_image = mask_depth_image(depth_image, depth_start, depth_end)
        #         # return mvs input
        #         self.counter += 1
        #         duration = time.time() - start_time
        #         images = np.stack(images, axis=0)
        #         cams = np.stack(cams, axis=0)
        #         print('range: ', cams[0, 1, 3, 0], cams[0, 1, 3, 1], cams[0, 1, 3, 2],
        #               cams[0, 1, 3, 3])
        #
        #         yield (images, cams, depth_image)

def valiation(traning_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)
    # if FLAGS.regularization == 'GRU':
    training_sample_size = training_sample_size
    print('sample number: ', training_sample_size)
    with  tf.device('/cpu:0'):
        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()
        ########## optimization options ##########
        global less_one_accuracy, less_three_accuracy, summaries, loss

        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images, cams, depth_images = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_images.set_shape(tf.TensorShape([None,FLAGS.view_num, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_num = tf.cast(
                        tf.reshape(tf.slice(cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), "int32")
                    depth_end = depth_start + (tf.cast(FLAGS.max_d, tf.float32) - 1) * depth_interval
                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True
                    ref_image = tf.squeeze(
                        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                    depth_image = tf.squeeze(
                        tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
                    # inference
                    if FLAGS.regularization == '1DCNNs':
                        # initial depth map
                        # depth_interval=depth_interval*2.0
                        loss, depth_map, less_one_accuracy, less_three_accuracy = inference_3(
                            images, cams, depth_image)
                        # depth_map, _ = inference_mem_6(
                        #     images, cams, FLAGS.max_d, depth_start, depth_end, is_master_gpu)
                        # loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #     depth_map, depth_image, depth_interval)
                    if FLAGS.regularization == 'GRU':
                        # initial depth map
                        # depth_interval=depth_interval*2.0
                        prob_volume = inference_1(
                            images, cams, FLAGS.max_d, depth_start, depth_end, is_master_gpu)
                        # loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #     depth_map, depth_image, depth_interval)
                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                        # depth_map,_= inference_winner_take_all(
                        #     images, cams, FLAGS.max_d, depth_start, depth_end, is_master_gpu)
                        # prob=tf.cast(prob_volume>0.5,tf.float32)
                        # prob=tf.reduce_mean(tf.reshape(prob_volume,[-1,64,FLAGS.max_h/4*FLAGS.max_w/4]),-1)#b,32
                        # depth_nums=(prob*256.0) # b,128
                        # depth_intervals=[]
                        # depth_interval=tf.reshape(depth_interval,[FLAGS.batch_size,1])#b,1
                        # probs=[]
                        # for idx in range(64):
                        #     depth_num=tf.ceil(tf.reshape(tf.slice(depth_nums,[0,idx],[-1,1]),[-1,1]))
                        #     probs.append(tf.tile(tf.slice(prob,[0,idx],[-1,1]),[1,tf.reshape(tf.cast(depth_num,tf.int32),[])]))
                        #     depth_intervals.append(tf.tile(depth_interval/(depth_num),[1,tf.reshape(tf.cast(depth_num,tf.int32),[])]))
                        # depth_intervals=tf.concat(depth_intervals,-1)
                        # # depth_intervals=tf.reshape(tf.tile(tf.reshape(depth_intervals,[-1,128,1]),[1,1,2]),[FLAGS.batch_size,256])
                        # # depth_intervals=tf.slice(depth_intervals,[0,0],[-1,128])
                        # depth_num=tf.shape(depth_intervals)[1]
                        # probs=tf.concat(probs,-1)
                        # depth_map,prob_map=inference_mem_5(ref_tower,view_towers ,cams,depth_num,depth_start,depth_intervals,probs)

                        # depth_map,prob_map=inference_mem_4(images,cams,FLAGS.max_d,depth_start,depth_end)
                        # loss=non_zero_mean_absolute_diff(depth_image, depth_map, tf.abs(depth_interval))
                        #mask=tf.cast(prob_map>0.3,dtype=tf.float32)
                        #depth_image=depth_image*mask
                        # if FLAGS.inverse_depth:
                        #     depth_start = tf.reshape(tf.div(1.0, depth_start), [])
                        #     depth_end = tf.reshape(tf.div(1.0, depth_end), [])
                        #     depth_interval=(depth_end-depth_start)/(FLAGS.max_d-1)
                        #
                        #     # mask = tf.cast(tf.not_equal(depth_image, 0.0), dtype="float32")
                        #     # none_zero = tf.cast(tf.count_nonzero(mask), dtype=tf.float32)+1e-7
                        #     # tf.where(mask,1.0/depth_image,0.0)
                        #
                        #     zeros = tf.zeros([FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,1])
                        #     depth_image=tf.where(depth_image>0.0,1.0/depth_image,zeros)
                        # loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #     depth_map, depth_image, depth_interval)
                        # mask = tf.cast(mask, dtype=tf.float32)
                        # depth_image = mask * depth_image
                        # loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                        #    mvsnet_classification_loss(
                        #        prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                        # depth_map,_ = inference_mem_1(images, cams, FLAGS.max_d,
                        #                                         depth_start, depth_end)
                        #depth_map=tf.expand_dims(depth_map,axis=-1)
                        # refinement
                        
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            refined_depth_map = depth_refine(depth_map, ref_image,
                                                             FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                            loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                                refined_depth_map, depth_image, depth_interval)
                            # loss = loss * 0.9 + loss1 * 0.1


        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        log=open('log_1.txt','a')
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            for index in range(100000,500000,5000):
                # load pre-trained model
                pretrained_model_path = os.path.join(FLAGS.model_dir, "19-07-22-8", FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                if os.path.exists('-'.join([pretrained_model_path, str(index)+".index"]))==False:
                    break
                restorer.restore(sess, '-'.join([pretrained_model_path, str(index)]))

                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_path, str(index)])), Notify.ENDC)
                less_one_accuracies=[]
                less_three_accuracies=[]
                output_dir="/home/haibao637/validation"
                if os.path.exists(output_dir)==False:
                    os.makedirs(output_dir)
                for epoch in range(1):
                    # training of one epoch
                    step = 0
                    sess.run(training_iterator.initializer)
                    for idx in range(int(training_sample_size / FLAGS.num_gpus)):
                        # run one batch
                        start_time = time.time()
                        try:
                            out_less_one, out_less_three = sess.run(
                                [less_one_accuracy, less_three_accuracy] )
                            less_one_accuracies.append(out_less_one)
                            less_three_accuracies.append(out_less_three)
                        except tf.errors.OutOfRangeError:
                            print("End of dataset")  # ==> "End of dataset"
                            break
                        duration = time.time() - start_time
                        # np.save(os.path.join(output_dir,"%08d"%idx),out_images[:,0,...])
                        # np.save(os.path.join(output_dir, "%08d_d" % idx), out_depth_map)
                        # print info
                        if step % FLAGS.display == 0:
                            print(Notify.INFO,
                                'index, %d, step %d, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                                (index, step, out_less_one, out_less_three, duration), Notify.ENDC)
                        # write summary
                        step += FLAGS.batch_size * FLAGS.num_gpus
                        total_step += FLAGS.batch_size * FLAGS.num_gpus

                print(Notify.INFO,
                    'validation  %d,  (< 1px) = %.4f, (< 3px) = %.4f' %
                    (index, sum(less_one_accuracies)/len(less_one_accuracies), sum(less_three_accuracies)/len(less_three_accuracies)), Notify.ENDC)
                log.writelines('%d,%.4f,%.4f\n'%(index, sum(less_one_accuracies)/len(less_one_accuracies), sum(less_three_accuracies)/len(less_three_accuracies)))
                acc1=sum(less_one_accuracies)/len(less_one_accuracies)
                acc3=sum(less_three_accuracies)/len(less_three_accuracies)
                vis.line(X=np.column_stack([index,index]),Y=np.column_stack([acc1,acc3]),
                win="mvsnet_accuracy",update='append',opts=dict(showlegend=True,legend=["accuracy_1","accuracy_3"]))

def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root,mode='validation')
    # Shuffle
    # random.shuffle(sample_list)
    # Training entrance.
    # pretrained_model_path = os.path.join(FLAGS.model_dir, "19-07-22-8", FLAGS.regularization, 'model.ckpt')
    # print('-'.join([pretrained_model_path, str(5000)]))
    valiation(sample_list)


if __name__ == '__main__':
    print('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

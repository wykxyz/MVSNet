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

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import *
from homography_warping import get_homographies, homography_warping

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# paths
tf.app.flags.DEFINE_string('dtu_data_root', '/home/yanjianfeng/data5//dtu/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('log_dir', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time = time.strftime("%y-%m-%d")
tf.app.flags.DEFINE_string('log_dir', os.path.join('/home/yanjianfeng/data6/tf_log/', time.strftime("%y-%m-%d"),
                                                   time.strftime("%H:%M:%S")),
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/yanjianfeng/data6/tf_models/'),
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step',75000,
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
tf.app.flags.DEFINE_string('regularization', '1DCNNs',
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
                for view in range(self.view_num):
                    image = center_image(cv2.imread(data[2 * view]))
                    cam = load_cam(open(data[2 * view + 1]))
                    cam[1][3][1] = cam[1][3][1] * (FLAGS.interval_scale)
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))
                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)
                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image)

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
                    images, cams, depth_image = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_end = depth_start + (tf.cast(FLAGS.max_d, tf.float32) - 1) * depth_interval
                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True
                    ref_image = tf.squeeze(
                        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                    # inference
                    if FLAGS.regularization == '1DCNNs':
                        # initial depth map
                        
                        prob_volume = inference(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        # loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #     depth_map, depth_image, depth_interval)
                        # mask = tf.cast(mask, dtype=tf.float32)
                        # depth_image = mask * depth_image
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                           mvsnet_classification_loss(
                               prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                        #depth_map,_ = inference_mem_1(images, cams, FLAGS.max_d,
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
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            # load pre-trained model
            pretrained_model_path = os.path.join(FLAGS.model_dir, "19-04-30", FLAGS.regularization, 'model.ckpt')
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))

            print(Notify.INFO, 'Pre-trained model restored from %s' %
                  ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            less_one_accuracies=[]
            less_three_accuracies=[]
            for epoch in range(1):
                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size / FLAGS.num_gpus)):
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
                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                              (epoch, step, out_less_one, out_less_three, duration), Notify.ENDC)
                    # write summary
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus
            print(Notify.INFO,
                  'validation  %d,  (< 1px) = %.4f, (< 3px) = %.4f' %
                  (FLAGS.ckpt_step, sum(less_one_accuracies)/len(less_one_accuracies), sum(less_three_accuracies)/len(less_three_accuracies)), Notify.ENDC)


def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root,mode='validation')
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    valiation(sample_list)


if __name__ == '__main__':
    print('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import json
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

# import high_dim_filter_loader

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import *
from homography_warping import get_homographies, homography_warping

# custom_module = high_dim_filter_loader.custom_module
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# paths

tf.app.flags.DEFINE_string('dtu_data_root', '/home/haibao637/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('eth3d_data_root', '/home/yanjianfeng/data6//eth3d_training/eth3d/',
#                            """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('logdirs', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time = time.strftime("%y-%m-%d-8")
tf.app.flags.DEFINE_string('logdirs', os.path.join('/home/haibao637/data/tf_log/', time.strftime("%y-%m-%d"),
                                                   time.strftime("%H:%M:%S")),
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/haibao637/data/tf_models/'),
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', False,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', 50000,
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
tf.app.flags.DEFINE_integer('batch_size', 30,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 40,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 1e-3,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 10000,
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
                depth_images = []
                # img = cv2.imread(data.image[0])
                for view in range(self.view_num):
                    images.append(center_image(cv2.imread(data[view]["image"])))
                    cams.append(load_cam(open(data[view]["cam"])))

                    # depth_start=cams[0][1,3,0]
                    depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                    depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                    depth_image = load_pfm(open(data[0]["depth"]))
                    depth_image = mask_depth_image(depth_image, depth_start, depth_end)
                    depth_images.append(depth_image)
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
                # depth_image = load_pfm(open(data.depths[0]))
                # depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                # depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                # depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                depth_images = np.stack(depth_images, 0)
                # depth_images=depth_images[...,np.newaxis]
                # print(depth_images.shape)
                # print('Forward pass: d_min = %f, d_max = %f.' % \
                #       (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_images)

                # return backward mvs input for GRU
                # if FLAGS.regularization == 'GRU':
                #     self.counter += 1
                # start_time = time.time()
                # cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                # cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                # duration = time.time() - start_time
                # print('Back pass: d_min = %f, d_max = %f.' % \
                #       (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                # yield (images, cams, depth_images)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(traning_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)
    # if FLAGS.regularization == 'GRU':
    training_sample_size = training_sample_size
    print('sample number: ', training_sample_size)

    with tf.Graph().as_default(), tf.device('/cpu:0'):

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
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images, cams, depth_images = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    # gray_image.set_shape(tf.TensorShape([None,None,None,1]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_images.set_shape(tf.TensorShape([None, None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_end = depth_start + depth_interval * (FLAGS.max_d - 1)
                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True
                    depth_image = tf.squeeze(
                        tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
                    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1]), 1)
                    ref_depth = tf.squeeze(tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1]), 1)
                    mask = tf.ones_like(depth_image, dtype=tf.bool)
                    for view in range(1, FLAGS.view_num):
                        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), 1)
                        view_depth = tf.squeeze(tf.slice(depth_images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), 1)
                        warp_depth, grid_depth, m = reprojection_depth(input_image=view_depth, left_cam=ref_cam,
                                                                       right_cam=view_cam, depth_map=ref_depth)
                        mask = tf.less_equal(tf.abs(warp_depth - grid_depth), tf.abs(depth_interval)) & mask & m

                    mask = tf.cast(mask, tf.float32)


        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.set_per_process_memory_growth()
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            # training several epochs


            # training of one epoch
            step = 0
            valid_list=[]
            sess.run(training_iterator.initializer)
            for index in range(0,30,FLAGS.batch_size):
                # run one batch
                start_time = time.time()
                try:
                    print("run %04f\n"%(index*1.0/training_sample_size))
                    out_mask = sess.run(mask)
                    for i in range(min(training_sample_size-index,FLAGS.batch_size)):
                        if(np.count_nonzero(out_mask[i])>200):
                            valid_list.append(traning_list[index+i])
                    # if(np.count_nonzero(out_mask)>200):
                    #     valid_list.append(traning_list[index])
                except tf.errors.OutOfRangeError:
                    print("End of dataset")  # ==> "End of dataset"
                    break
            file = open('log.txt', 'w')
            from json import  JSONEncoder
            class Myencoder(JSONEncoder):
                def default(self,o):
                    return o.__dict__
            myencode=Myencoder()
            json.dumps(sample_list,open("data.json",'w'),cls=Myencoder)

        file.close()


def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    # sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root)
    # sample_list1=gen_eth3d_path(FLAGS.eth3d_data_root)
    # sample_list.extend(sample_list1)
    # Shuffle
    # random.shuffle(sample_list)
    # # Training entrance.
    # train(sample_list)
    pf=open("pair.txt",'r')
    sample_list=json.load(pf)
    train(sample_list)

if __name__ == '__main__':
    print('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

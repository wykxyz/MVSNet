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
from flowmodel import *
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# paths

tf.app.flags.DEFINE_string('dtu_data_root', '/home/haibao637/data/mvs_training/dtu/',
                           """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('eth3d_data_root', '/home/yanjianfeng/data6//eth3d_training/eth3d/',
#                            """Path to dtu dataset.""")
# tf.app.flags.DEFINE_string('logdirs', '/home/yanjianfeng/data5/tf_log',
#                            """Path to store the log.""")
# tf.app.flags.DEFINE_string('model_dir', '/home/yanjianfeng/data5/tf_model',
#                            """Path to save the model.""")
train_time=time.strftime("%y-%m-%d-8")
tf.app.flags.DEFINE_string('logdirs', os.path.join('/home/haibao637/data/tf_log/',time.strftime("%y-%m-%d"),time.strftime("%H:%M:%S")),
						   """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join('/home/haibao637/data/tf_models/') ,
						   """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', False,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', 400000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 160,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 128,
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
tf.app.flags.DEFINE_integer('num_gpus',1,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
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
tf.app.flags.DEFINE_bool('inverse_depth', True,
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
                    img=center_image(cv2.imread(data[view].image))
                    images.append(cv2.resize(img,(img.shape[1]/4,img.shape[0]/4)))
                    cams.append(load_cam(open(data[view].cam)))

                # depth_start=cams[0][1,3,0]
                    depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                    depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                    depth_image = load_pfm(open(data[view].depth))
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
                depth_images=np.stack(depth_images,0)
                # depth_images=depth_images[...,np.newaxis]
                # print(depth_images.shape)
                print('Forward pass: d_min = %f, d_max = %f.' % \
                   (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_images)

                # return backward mvs input for GRU
                # if FLAGS.regularization == 'GRU':
                #     self.counter += 1
                # start_time = time.time()
                cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                duration = time.time() - start_time
                print('Back pass: d_min = %f, d_max = %f.' % \
                   (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                yield (images, cams, depth_images)

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
            if g is None :
                continue
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
    training_sample_size = len(traning_list)*2
    #if FLAGS.regularization == 'GRU':
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

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step,
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        opt = tf.train.AdamOptimizer(learning_rate=lr_op)

        global less_one_accuracy, less_three_accuracy,summaries,loss
        tower_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images, cams, depth_images = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    # gray_image.set_shape(tf.TensorShape([None,None,None,1]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_images.set_shape(tf.TensorShape([None,None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_end=depth_start+depth_interval*(FLAGS.max_d-1)
                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True
                    depth_image = tf.squeeze(
                        tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
                    ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
                    ref_depth = tf.squeeze(tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1]),1)
                    mask=tf.ones_like(depth_image,dtype=tf.float32)
                    for view in range(1,FLAGS.view_num):
                        view_cam=tf.squeeze(tf.slice(cams,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
                        view_depth=tf.squeeze(tf.slice(depth_images,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
                        warp_depth,grid_depth,m=reprojection_depth(input_image=view_depth,left_cam=ref_cam,right_cam=view_cam,depth_map=ref_depth)
                        mask=tf.cast(tf.less_equal(tf.abs(warp_depth-grid_depth),tf.abs(depth_interval))&m,tf.float32)+mask

                    mask=tf.cast(tf.greater_equal(mask,2.0),tf.float32)
                    depth_image=depth_image*mask
                    # inference
                    # if FLAGS.regularization == '1DCNNs':
                        # depth_image=tf.squeeze(
                        #      tf.slice(depth_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 1]), axis=1)
                        # initial depth map
                        # loss,depth_map,less_one_accuracy,less_three_accuracy = inference_3(
                        #     images, cams,depth_images)
                        # with tf.name_scope("crf_layers"):
                        #     # prob_volume=tf.stop_gradient(prob_volume,'stop_gradient')
                        #     prob_volume = tf.transpose(tf.squeeze(prob_volume, -1), (0, 2, 3, 1))  # b,h,w,d
                        #     # unary=-prob_volume
                        #     x, y = tf.meshgrid(np.arange(0, 1.0, 1.0 / (FLAGS.max_w / 4)),
                        #                        np.arange(0, 1.0, 1.0 / (FLAGS.max_h / 4)))
                        #
                        #     coords = tf.cast(tf.expand_dims(tf.stack([x, y], -1), 0) - 0.5,tf.float32)
                        #     ref_image=tf.image.resize_bilinear(ref_image,[FLAGS.max_h/4, FLAGS.max_w/4])
                        #     ref_feature = tf.concat((ref_image, coords), axis=-1)  # b,h,w,rgb+coord
                        #     crf_prob_volume = crf_layer(prob_volume, ref_feature)  # h,w,w,d
                        #     crf_prob_volume = tf.expand_dims(tf.transpose(crf_prob_volume, (0, 3, 1, 2)), axis=-1)
                        #     crf_prob_volume=tf.nn.softmax(crf_prob_volume,axis=1)
                        # loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #     depth_map, depth_image, depth_interval)
                        # volume_shape = tf.shape(prob_volume)
                        # if FLAGS.inverse_depth:
                        #     inv_depth_start = tf.reshape(tf.div(1.0, depth_start),[])
                        #     inv_depth_end = tf.reshape(tf.div(1.0, depth_end),[])
                        #     # inv_depth_interval = (inv_depth_start - inv_depth_end) / (tf.cast(FLAGS.max_d, 'float32') - 1)
                        #     inv_depth = tf.reshape(tf.lin_space(inv_depth_start, inv_depth_end, FLAGS.max_d),[-1,FLAGS.max_d,1,1,1])
                        #     inv_depth=tf.reduce_sum(inv_depth*prob_volume,axis=1)#b,h,w,1
                        #
                        #     depth_map = tf.div(1.0, inv_depth)
                        #     loss,less_one_accuracy,less_three_accuracy=mvsnet_regression_loss(depth_map,depth_image,depth_interval)
                        # else:
                        # with tf.name_scope("crf"):
                            # loss,less_one_accuracy,less_three_accuracy=mvsnet_crf_loss(prob_volume,gray_image,depth_image,FLAGS.max_d,depth_start,depth_interval)



                            # inv_depth = inv_depth_start - d_idx * inv_interval
                            # depth = tf.div(1.0, inv_depth)
                        # if FLAGS.inverse_depth:
                            # depth_start = tf.reshape(tf.div(1.0, depth_start), [])
                            # depth_end = tf.reshape(tf.div(1.0, depth_end), [])
                            # depth_interval=(depth_end-depth_start)/(FLAGS.max_d-1)

                            # mask = tf.cast(tf.not_equal(depth_image, 0.0), dtype="float32")
                            # none_zero = tf.cast(tf.count_nonzero(mask), dtype=tf.float32)+1e-7
                            # tf.where(mask,1.0/depth_image,0.0)

                            # zeros = tf.zeros([FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,1])
                            # depth_image=tf.where(depth_image>0.0,1.0/depth_image,zeros)

                        # loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                        #     mvsnet_classification_loss(
                        #         prob_volume, inv_depth_image, FLAGS.max_d, inv_depth_end, inv_depth_interval)
                        # lines = tf.cast(tf.reshape(tf.linspace(0.0, FLAGS.max_d*1.0 - 1, FLAGS.max_d), [1, FLAGS.max_d]),dtype=tf.float32)  # d
                        # regression_inv_depth_base = tf.tile(tf.reshape(tf.reshape(inv_depth_start, [-1, 1])
                        #                                                + tf.reshape(inv_depth_interval,
                        #                                                             [-1, 1]) * lines,
                        #                                                [-1, FLAGS.max_d,1, 1,1]),
                        #                                     [1,1, FLAGS.max_h / 4, FLAGS.max_w / 4,1])  # b,d,h,w,1
                        # regression_inv_depth_map = tf.reduce_mean(regression_inv_depth_base * prob_volume, axis=1)#b,h,w,1
                        # regression_depth_map = 1.0 / (regression_inv_depth_map + 1e-7)
                        # loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                        #         mvsnet_classification_loss(
                        #             prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                        # _,less_one_accuracy,less_three_accuracy=mvsnet_regression_loss(depth_map,depth_image,depth_interval)
                        # loss3 = tf.reduce_sum(mask * ((depth_image - regression_depth_map)/tf.reshape(depth_interval,[-1,1,1,1])) ** 2) / none_zero
                        # depth_map=regression_depth_map
                        # mask=depth_image>0
                        # none_zero=tf.cast(tf.count_nonzero(mask),tf.float32)
                        # mask=tf.cast(mask,dtype=tf.float32)
                        # expand_depth_map=tf.extract_image_patches(images=depth_map, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                        #                          rates=[1, 1, 1, 1], padding='SAME')#b,h,w,9
                        # expand_gt_map=tf.extract_image_patches(images=depth_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                        #                          rates=[1, 1, 1, 1], padding='SAME')#b,h,w,9
                        # Ed=expand_depth_map-tf.reduce_mean(expand_depth_map,axis=-1,keepdims=True)
                        # Eg=expand_gt_map-tf.reduce_mean(expand_gt_map,axis=-1,keepdims=True)
                        # sig=tf.sqrt(tf.reduce_mean(Ed**2,axis=-1,keepdims=True)*tf.reduce_mean(Eg**2,axis=-1,keepdims=True))+1e-7
                        # loss1=tf.reduce_sum(mask*tf.reduce_mean(1-Ed*Eg/sig,axis=-1,keepdims=True))/(none_zero+1e-7)
                        # loss2=tf.reduce_sum(tf.reduce_mean(2*(expand_depth_map-depth_map)/(expand_depth_map+depth_map+1e-7),axis=-1,keepdims=True))/(none_zero+1e-7)
                        # loss=loss+loss1+loss2
                        # def E(input):
                        #     return input
                        # loss=E((expand_depth_map-E(expand_depth_map))*(expand_gt_map-E(expand_gt_map)))/\
                        #     sqrt(E((expand_depth_map-E(expand_depth_map))**2)*E((expand_gt_map-E(expand_gt_map))**2))
                        # loss=E(2*(expand_depth_map-depth_map)/(expand_depth_map+depth_map))
                        # refinement
                        # if FLAGS.refinement:
                        #     ref_image = tf.squeeze(
                        #          tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                        #     refined_depth_map = depth_refine(depth_map, ref_image,
                        #                                       FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        #     loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                        #          refined_depth_map, depth_image, depth_interval)
                        #     loss=loss*0.5+loss1*0.5






                    if FLAGS.regularization == 'GRU':

                        # probability volume
                        # prob_volume = inference_prob_recurrent_1(
                        #     images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        depth_map=depth_inference(images,cams)

                        

                        # if FLAGS.inverse_depth:
                        #     inv_depth_start = tf.reshape(tf.div(1.0, depth_start),[])
                        #     inv_depth_end = tf.reshape(tf.div(1.0, depth_end),[])
                        #     # inv_depth_interval = (inv_depth_start - inv_depth_end) / (tf.cast(FLAGS.max_d, 'float32') - 1)
                        #     inv_depth = tf.reshape(tf.lin_space(inv_depth_start, inv_depth_end, FLAGS.max_d),[-1,FLAGS.max_d,1,1,1])
                        #     inv_depth=tf.reduce_sum(inv_depth*prob_volume,axis=1)#b,h,w,1
                        #     depth_map = tf.div(1.0, inv_depth)
                        # else:
                        #     depth_map=tf.reshape(tf.linspace(depth_start,depth_end,FLAGS.max_d),[-1,FLAGS.max_d,1,1,1,])
                        #     depth_map=tf.reduce_sum(depth_map*prob_volume,axis=1)

                        loss,less_one_accuracy,less_three_accuracy=mvsnet_regression_loss(depth_map,depth_image,depth_interval)
                        # K=tf.reshape(tf.slice(cams,[0,0,1,0,0],[-1,1,1,3,3]),[-1,3,3])
                        # loss_1=normal_loss(depth_map,depth_image,K)
                        # loss+=loss_1
                      
                        # classification loss
                        # loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                        #     mvsnet_classification_loss(
                        #         prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    # retain the summaries from the final tower.

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # average gradient
        # grads = average_gradients(tower_grads)

        # training opt
        # train_opt=tf.cond(loss>0,lambda:opt.apply_gradients(grads, global_step=global_step),lambda:loss)
        train_opt = opt.apply_gradients(grads, global_step=global_step)

        # summary
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('less_one_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_accuracy', less_three_accuracy))
        summaries.append(tf.summary.scalar('lr', lr_op))
        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
      
        config.gpu_options.allow_growth = True
        # config.gpu_options.set_per_process_memory_growth()
        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(FLAGS.logdirs, sess.graph)

            # load pre-trained model
            if FLAGS.use_pretrain:
                pretrained_model_path = os.path.join(FLAGS.model_dir,"19-07-26-8", FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
                # pretrained_model_path= '-'.join(["/home/haibao637/data5/model/tf_model/3DCNNs/model.ckpt", str(FLAGS.ckpt_step)])
                # restorer.restore (sess,pretrained_model_path )
                print(Notify.INFO, 'Pre-trained model restored from %s'%pretrained_model_path, Notify.ENDC)
                # global_step=tf.assign(global_step,0)
                total_step =FLAGS.ckpt_step

            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0

                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size / FLAGS.num_gpus)):

                    # run one batch
                    start_time = time.time()
                    try:
                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                            [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time
                    # print(out_depth.min(),out_depth.max())
                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                              (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)

                    # write summary
                    if (total_step % (FLAGS.display * 10) == 0) and  (out_loss>0):
                        summary_writer.add_summary(out_summary_op, total_step)

                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(FLAGS.model_dir,train_time, FLAGS.regularization)
                        if not os.path.exists(model_folder):
                            os.makedirs(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus
def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root)
    # sample_list1=gen_eth3d_path(FLAGS.eth3d_data_root)
    # sample_list.extend(sample_list1)
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    train(sample_list)


if __name__ == '__main__':
    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

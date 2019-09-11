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

# paths
tf.app.flags.DEFINE_string('dtu_data_root', '/xdata/wuyk/mvs_training/dtu/', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('log_dirs', '/xdata/wuyk/tf_data4-validate/tf_log', #tf1.14: log_dir define twice
                           """path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', '/xdata/wuyk/tf_data4-validate/tf_model',
                           """path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True, 
                            """whether to train.""")
tf.app.flags.DEFINE_string('mode', 'validation', 
                            """whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', 0,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 5, 
                            """number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256, 
                            """maximum depth step when training.""") # depth num
tf.app.flags.DEFINE_integer('max_w', 640, 
                            """maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 512, 
                            """maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.06, 
                            """downsample scale for building cost volume.""")

# network architectures
tf.app.flags.DEFINE_string('regularization', 'gru',
                           """regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                           """Whether to apply inverse depth for R-MVSNet""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 1, 
                            """Training epoch number.""")
# tf.app.flags.DEFINE_integer('num_cpu_core', 4,
#                           """num cpu core.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")


FLAGS = tf.app.flags.FLAGS
print('initial FLAGS---------------------------------------------\n')
print(FLAGS)

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
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
                    cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale # origin para * interval_scale
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num]))

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1] # interval
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1] #define max depth
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                print('Forward pass: d_min = %f, d_max = %f.' % \
                    (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1])) # 191*2.5*1.06 + 425
                yield (images, cams, depth_image) 

                # return backward mvs input for GRU
                if FLAGS.regularization == 'GRU':
                    self.counter += 1
                    start_time = time.time()
                    cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                    cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                    duration = time.time() - start_time
                    print('Back pass: d_min = %f, d_max = %f.' % \
                        (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                    yield (images, cams, depth_image) 

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
            #print(g)
            #print(g.shape)
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
    # if FLAGS.regularization == 'GRU': #Why
    #     training_sample_size = training_sample_size * 2
    # print ('sample number: ', training_sample_size)

    if os.path.exists(FLAGS.log_dirs):
        print('remove ', FLAGS.log_dirs)
        os.rmdir(FLAGS.log_dirs) #os.removedirs
    print('create ', FLAGS.log_dirs)
    os.makedirs(FLAGS.log_dirs)
    
    if not os.path.exists(FLAGS.model_dir):
        print('create ', FLAGS.model_dir)
        os.makedirs(FLAGS.model_dir)

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

        tower_maes = []
        for i in xrange(FLAGS.num_gpus):
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
                    #depth_interval_numpy = depth_interval.eval()
                    print('depth_interval: ', depth_interval)
                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    # inference
                    if FLAGS.regularization == '3DCNNs':

                        # initial depth map
                        depth_map, prob_map = inference(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # refinement
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            refined_depth_map = depth_refine(depth_map, ref_image, 
                                    FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        else:
                            refined_depth_map = depth_map

                        # regression loss
                        loss0, less_one_temp, less_three_temp = mvsnet_regression_loss(
                            depth_map, depth_image, depth_interval)
                        loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                            refined_depth_map, depth_image, depth_interval)
                        loss = (loss0 + loss1) / 2
                        mae = loss1 
                    elif FLAGS.regularization == '3DCNNs_W':

                        # initial depth map
                        depth_map, prob_map = inference_cos_weighted(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # refinement
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            refined_depth_map = depth_refine(depth_map, ref_image,
                                    FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        else:
                            refined_depth_map = depth_map

                        # regression loss
                        loss0, less_one_temp, less_three_temp = mvsnet_regression_loss(
                            depth_map, depth_image, depth_interval)
                        loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                            refined_depth_map, depth_image, depth_interval)
                        loss = (loss0 + loss1) / 2
                        mae = loss1 

                    elif FLAGS.regularization == 'GRU':

                        # probability volume
                        #prob_volume = inference_prob_recurrent_wori(
                        #    images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        prob_volume = inference_prob_recurrent(
                           images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                        

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    elif FLAGS.regularization == 'GRU_WORI':

                        # probability volume
                        prob_volume = inference_prob_recurrent_wori(
                           images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)                        

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    elif FLAGS.regularization == 'GRU_W':

                        # probability volume
                        prob_volume = inference_prob_recurrent_w(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    elif FLAGS.regularization == 'GRU_WGATE':

                        # probability volume
                        prob_volume = inference_prob_recurrent_wgate(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    elif FLAGS.regularization == 'GRU_WGATECOS':

                        # probability volume
                        prob_volume = inference_prob_recurrent_wgatecos(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    elif FLAGS.regularization == 'GRU_NONLOCALVIEWNUM':

                        # probability volume
                        prob_volume = inference_prob_recurrent_nonlocalviewnum(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    
                    elif FLAGS.regularization == 'GRU_NONLOCALHW':

                        # probability volume
                        prob_volume = inference_prob_recurrent_nonlocalHW(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    # retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    tower_maes.append(mae)

        #if FLAGS.regularization not in ['3DCNNs_W', '3DCNNs']:
        tower_maes = tf.concat(axis=0, values=tower_maes)
        average_maes = tf.reduce_mean(tower_maes)
        summaries.append(tf.summary.scalar('mae', mae))
        summaries.append(tf.summary.scalar('average_maes', average_maes))

        # summary
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('less_one_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_accuracy', less_three_accuracy))
        
        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)        
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        var_init_op = tf.local_variables_initializer()
        # GPU grows incrementally
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:     
            
            # initialization
            # initialization
            sess.run(var_init_op)
            sess.run(init_op)
            total_step = 0

            summary_writer = tf.summary.FileWriter(FLAGS.log_dirs, sess.graph)

            # load model
            if FLAGS.model_dir is not None:
                pretrained_model_ckpt_path = os.path.join(FLAGS.model_dir, FLAGS.regularization, 'model.ckpt') 
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step

            # Evaluate of one epoch
            epoch = FLAGS.epoch
            step = 0
            #TODO: define mean averge error
            total_average_mae = 0
            sess.run(training_iterator.initializer)
            for _ in range(int(training_sample_size / FLAGS.num_gpus)):

                # run one batch
                start_time = time.time()
                try:
                    out_summary_op, out_average_maes, out_loss, out_less_one, out_less_three = sess.run(
                    [summary_op, average_maes, loss, less_one_accuracy, less_three_accuracy])
                except tf.errors.OutOfRangeError:
                    print("End of dataset")  # ==> "End of dataset"
                    break
                duration = time.time() - start_time

                total_average_mae += out_average_maes
                # print info
                if step % FLAGS.display == 0:
                    print(Notify.INFO,
                        'epoch, %d, step %d, total_step %d, total_average_mae = %.4f, out_average_maes = %.4f, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)'
                        % (epoch, step, total_step, total_average_mae, out_average_maes,out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                
                # write summary
                if step % (FLAGS.display * 10) == 0:
                    summary_writer.add_summary(out_summary_op, total_step)
                
                step += FLAGS.batch_size * FLAGS.num_gpus
                total_step += FLAGS.batch_size * FLAGS.num_gpus

            mean_total_average_mae = total_average_mae / step
            print(Notify.INFO,'step %d, total_step %d, total_average_mae = %.4f, mean_total_average_mae = %.4f' % 
                (step, total_step, total_average_mae, mean_total_average_mae), Notify.ENDC)

def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root, FLAGS.mode)
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    print('Change FLAGS*********************************************\n')
    print(FLAGS)
    train(sample_list)


if __name__ == '__main__':
    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

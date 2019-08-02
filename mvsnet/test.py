#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse

import imageio
import numpy as np

import cv2
import tensorflow as tf

from utils import depth, best_depth_range

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from loss import *
# import pydensecrf.densecrf as dcrf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# dataset parameters
tf.app.flags.DEFINE_string('dense_folder',"/home/haibao637/data/tankandtemples/intermediate/Family/",
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('model_dir', 
                           '/home/haibao637//tf_model/',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 100000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('max_w', 1920,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('max_h', 1072,
                            """Maximum image height when testing.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 0.8,
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_float('base_image_size', 8, 
                            """Base image size""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Testing batch size.""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True, 
                            """Let image size to fit the network, including 'scaling', 'cropping'""")

# network architecture
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method, including '3DCNNs' and 'GRU'""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                           """Whether to apply inverse depth for R-MVSNet""")

FLAGS = tf.app.flags.FLAGS

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
                
                # read input data
                images = []
                cams = []
                image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
                print(image_index)
                selected_view_num = int(len(data) / 2)

                for view in range(min(self.view_num, selected_view_num)):
                    # image_file = file_io.FileIO(data[2 * view], mode='r')
                    # image = scipy.misc.imread(image_file, mode='RGB')
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image=cv2.imread(data[2 * view])
                    # cam_file = file_io.FileIO(data[2 * view + 1], mode='r')
                    cam = load_cam(open(data[2 * view + 1]), FLAGS.interval_scale)
                    if cam[1][3][2] == 0:
                        cam[1][3][2] = FLAGS.max_d
                    images.append(image)
                    cams.append(cam)

                if selected_view_num < self.view_num:
                    for view in range(selected_view_num, self.view_num):
                        # image_file = file_io.FileIO(data[0], mode='r')
                        # image = scipy.misc.imread(image_file, mode='RGB')
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image = cv2.imread(data[0])
                        # cam_file = file_io.FileIO(data[2 * view + 1], mode='r')
                        cam = load_cam(open(data[1]), FLAGS.interval_scale)
                        # cam_file = file_io.FileIO(data[1], mode='r')
                        # cam = load_cam(cam_file, FLAGS.interval_scale)
                        images.append(image)
                        cams.append(cam)



                # determine a proper scale to resize input 
                resize_scale = 1
                if FLAGS.adaptive_scaling:
                    h_scale = 0
                    w_scale = 0
                    for view in range(self.view_num):
                        height_scale = float(FLAGS.max_h) / images[view].shape[0]
                        width_scale = float(FLAGS.max_w) / images[view].shape[1]
                        if height_scale > h_scale:
                            h_scale = height_scale
                        if width_scale > w_scale:
                            w_scale = width_scale
                    if h_scale > 1 or w_scale > 1:
                        print ("max_h, max_w should < W and H!")
                        exit(-1)
                    resize_scale = h_scale
                    if w_scale > h_scale:
                        resize_scale = w_scale
                scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

                # crop to fit network
                croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)

                # real_cams=scaled_cams*1.0

                # center images
                centered_images = []
                for view in range(self.view_num):
                    centered_images.append(center_image(croped_images[view]))
                centered_images=np.stack(centered_images,axis=0)
                croped_cams=np.stack(croped_cams,axis=0)
                # sample cameras for building cost volume
                real_cams = np.copy(croped_cams) 
                scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

                # return mvs input
                scaled_images = []
                for view in range(self.view_num):
                    scaled_images.append(scale_image(croped_images[view], scale=FLAGS.sample_scale))
                scaled_images = np.stack(scaled_images, axis=0)
                croped_images = np.stack(croped_images, axis=0)
                real_images = croped_images * 1.0
                scaled_cams = np.stack(scaled_cams, axis=0)

                # depths = []
                # k1 = scaled_cams[0, 1, :3, :3]
                # p1 = scaled_cams[0, 0, :, :]
                # borders = [[0.0, 0.0, 1.0], [0.0, FLAGS.max_h / 4, 1.0], [FLAGS.max_w / 4, FLAGS.max_h / 4, 1.0],
                #            [FLAGS.max_w / 4, 0, 1.0]]
                # for view in range(self.view_num):
                #     k2 = scaled_cams[view, 1, :3, :3]
                #     p2 = scaled_cams[view, 0, :, :]
                #     depths.extend([depth(x+np.random.rand(1), x0+np.random.rand(1), k1, k2, p1, p2) for x in borders for x0 in borders])
                # depths = np.array(depths)
                # depths[depths < 0] = 0
                # dmax = depths.max()
                # dmin = max(0.3,depths[depths > depths.min()].min())


                # scaled_cams[0, 1, 3, 2] = int(2.0/0.002/4)*4
                # scaled_cams[0, 1, 3, 1] = 0.0025
                # scaled_cams[0, 1, 3, 3]=dmax
                # print(dmin, dmax)
                #scaled_cams[0, 1, 3, 0] *= 0.8
                print('range: ', scaled_cams[0,1, 3, 0], scaled_cams[0,1, 3, 1], scaled_cams[0,1, 3, 2], scaled_cams[0,1, 3, 3])
                self.counter += 1
                image_index="%08d"%image_index
                yield (scaled_images, centered_images, scaled_cams, real_images,real_cams)

def mvsnet_pipeline(mvs_list):

    """ mvsnet in altizure pipeline """
    print ('sample number: ', len(mvs_list))

    # create output folder
    output_folder = os.path.join(FLAGS.dense_folder, 'depths_mvsnet')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # testing set
        mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32,tf.float32,tf.float32)
        mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
        mvs_set = mvs_set.batch(FLAGS.batch_size)
        mvs_set = mvs_set.prefetch(buffer_size=1)
        mvs_iterator = mvs_set.make_initializable_iterator()
        with tf.device("/gpu:0"), tf.name_scope('Model_tower0'):
            # data from dataset via iterator

            scaled_images, centered_images, scaled_cams, real_images,real_cams = mvs_iterator.get_next()

            # set shapes
            scaled_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
            centered_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
            scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
            depth_start = tf.reshape(
                tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
            depth_interval = tf.reshape(
                tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
            depth_num = tf.cast(
                tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), "int32")
            # depth_end=tf.reshape(
            #     tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
            # depth_interval=(depth_end-depth_start)/(FLAGS.max_d-1)
            # deal with inverse depth
            # if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
            # depth_end = tf.reshape(
            #     tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
            # else:
            depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
            depth_interval = (depth_end - depth_start) / (FLAGS.max_d - 1)
            # depth_interval=tf.constant(0.002)
            # depth_num=
            # depth_ranges=tf.linspace(0.001,0.5,num=FLAGS.max_d//2)
            # inv_depth_ranges=tf.linspace(tf.ones_like(tf.reshape(depth_end,[])*2),tf.reshape(1.0/depth_end,[]),num=FLAGS.max_d//2)
            # depths=tf.concat([depth_ranges,inv_depth_ranges],0)
            # depth map inference using 3DCNNs
            # if FLAGS.regularization == '1DCNNs':
            #     init_depth_map, prob_map = inference_mem_6(centered_images, scaled_cams, FLAGS.max_d,
            #                                                          depth_start, depth_end)
            # elif  FLAGS.regularization == 'GRU':
            init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,FLAGS.max_d,
                                                                    depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)
            # prob_volume = inference_1(
            #     centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_end,True)
#             depth_end=depth_interval*(FLAGS.max_d-1)+depth_start
            # prob_map=tf.reduce_max(prob_volume,axis=1)#b,h,w,1
            # prob_index=tf.cast(tf.argmax(prob_volume,axis=1),tf.float32)#b,h,w,1
            # if FLAGS.inverse_depth:
            #     inv_depth_start = tf.div(1.0, depth_start)
            #     inv_depth_end = tf.div(1.0, depth_end)
            #     inv_interval = (inv_depth_start - inv_depth_end) / (FLAGS.max_d - 1)
            #     inv_depth = inv_depth_start - prob_index * inv_interval
            #     init_depth_map = tf.div(1.0, inv_depth)
            # else:
            #     init_depth_map = depth_start + prob_index * depth_interval
                # ref_features,view_features=get_features(centered_images,True)
    #         with tf.device("/gpu:0"):
    #                 init_depth_maps=[]
    #                 prob_maps=[]
    #                 prob_sums=[]
    #                 for i in range(4):
    #                     with tf.device("/gpu:%d"%i):
    #                         depth_num=tf.cast(depth_num,tf.float32)
    #                         local_depth_start=depth_start+depth_interval*(depth_num/4*i)
    #                         local_depth_end=depth_start+depth_interval*(depth_num/4*(i+1))
    #                         init_depth_map,prob_map,prob_sum=inference_mem_2(centered_images, scaled_cams, depth_num/4, local_depth_start, local_depth_end)
    #                         init_depth_maps.append(init_depth_map)
    #                         prob_maps.append(prob_map)
    #                         prob_sums.append(prob_sum)
    #                 init_depth_maps=tf.stack(init_depth_maps,-1)
    #                 prob_maps = tf.stack(prob_maps, -1)
    #
    #                 prob_sums = tf.stack(prob_sums, -1)
    #
    #                 prob_sum=tf.reduce_sum(prob_sums,-1)
    #                 prob_map=tf.reduce_max(prob_maps,-1)
    #                 prob_map=prob_map/(prob_sum+1e-7)
    #                 prob_index=tf.argmax(prob_map,-1)
    #                 mask=tf.one_hot(prob_index,4,axis=-1,dtype=tf.float32)
    #                 init_depth_map=tf.reduce_sum(tf.squeeze(init_depth_maps,-2)*mask,axis=-1,keepdims=True)
                    # init_depth_map=tf.gather(tf.reshape(init_depth_maps,[-1,4]),tf.reshape(prob_index,[-1,4]),axis=-1)
                    # init_depth_map=tf.reshape(init_depth_map,[-1,FLAGS.max_h/4,FLAGS.max_w/4,1])


            #             prob_volume = inference(
    #                 centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)
    #             with tf.name_scope("crf_layers"):
    #                 # prob_volume=tf.stop_gradient(prob_volume,'stop_gradient')
    #                 prob_volume = tf.transpose(tf.squeeze(prob_volume, -1), (0, 2, 3, 1))  # b,h,w,d
    #                 # unary=-prob_volume
    #                 x, y = tf.meshgrid(np.arange(0, 1.0, 1.0 / (FLAGS.max_w / 4)),
    #                                    np.arange(0, 1.0, 1.0 / (FLAGS.max_h / 4)))
    #                 ref_image = tf.squeeze(
    #                     tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    #                 coords = tf.cast(tf.expand_dims(tf.stack([x, y], -1), 0) - 0.5, tf.float32)
    #                 ref_image = tf.image.resize_bilinear(ref_image, [FLAGS.max_h / 4, FLAGS.max_w / 4])
    #                 ref_feature = tf.concat((ref_image, coords), axis=-1)  # b,h,w,rgb+coord
    #                 crf_prob_volume = crf_layer(prob_volume, ref_feature)  # h,w,w,d
    #                 crf_prob_volume = tf.expand_dims(tf.transpose(crf_prob_volume, (0, 3, 1, 2)), axis=-1)
    #                 crf_prob_volume = tf.nn.softmax(crf_prob_volume, axis=1)
    #             if FLAGS.inverse_depth:
    #                 inv_depth_start = tf.reshape(tf.div(1.0, depth_start),[])
    #                 inv_depth_end = tf.reshape(tf.div(1.0, depth_end),[])
    #                 inv_depth_interval = (inv_depth_start - inv_depth_end) / (tf.cast(FLAGS.max_d, 'float32') - 1)
    #                 # inv_depth = tf.reshape(tf.lin_space(inv_depth_start, inv_depth_end, FLAGS.max_d),[-1,FLAGS.max_d,1,1,1])
    #                 # inv_depth=tf.reduce_sum(inv_depth*crf_prob_volume,axis=1)#b,h,w,1
    #                 inv_depth=inv_depth_start-tf.cast(tf.argmax(crf_prob_volume,axis=1),tf.float32)*inv_depth_interval
    #                 init_depth_map = tf.div(1.0, inv_depth)
    #                 prob_map=tf.reduce_max(crf_prob_volume,axis=1)
    #             else:
    #                 init_depth_map=depth_start+tf.cast(tf.argmax(crf_prob_volume,axis=1),tf.float32)*depth_interval
    #                 prob_map=tf.reduce_max(crf_prob_volume,axis=1)
                # prob_volume=inference(centered_images,scaled_cams,FLAGS.max_d,depth_start,depth_interval)

                # depth_map=tf.expand_dims(init_depth_map,axis=-1)
                # refinement
                # if FLAGS.refinement:
                #     ref_image = tf.squeeze(
                #         tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                #     refined_depth_map = depth_refine(depth_map, ref_image,
                #                                      FLAGS.max_d, depth_start, depth_interval)
            # with tf.device("/gpu:1"):
            #     init_depth_map,prob_map=getDepthmap(cost_volume, FLAGS.max_d, depth_start, depth_interval)
            # init_depth_map, prob_map = inference_mem(
            #     centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)

            # if FLAGS.refinement:
            #     ref_image = tf.squeeze(tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            #     refined_depth_map = depth_refine(
            #         init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, True)

            # depth map inference using GRU
            # elif FLAGS.regularization == 'GRU':
            #     init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,
            #         depth_num, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)

        # init option
        init_op = tf.global_variables_initializer()
        var_init_op = tf.local_variables_initializer()

        # GPU grows incrementally
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        tf.logging.set_verbosity(tf.logging.INFO)
        # initialization option
        with tf.Session(config=config) as sess:

            # initialization
            sess.run(var_init_op)
            sess.run(init_op)
            total_step = 0

            # load model
            if FLAGS.model_dir is not None:
                pretrained_model_ckpt_path = os.path.join(FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                      ('-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step

            # run inference for each reference view
            sess.run(mvs_iterator.initializer)
            for out_index in range(len(mvs_list)):

                start_time = time.time()
                try:
                    # _=sess.run(prob_volume)
                    # continue
                    out_init_depth_map, out_prob_map, out_images, out_cams = sess.run(
                        [init_depth_map, prob_map, scaled_images, scaled_cams])
                    # sess.run(depth_start)
                    # out_prob_volume, out_images, out_cams, out_index,out_start,out_interval = sess.run(
                    #         [prob_volume, real_images, real_cams, image_index,depth_start,depth_interval])
                    # print("Running DenseCRF...")
                    # # unary_energy=np.max(out_prob_volume,axis=1)
                    # # unary_energy = out_prob_volume
                    # unary_energy=np.squeeze(out_prob_volume)#d,h,w
                    # crf = dcrf.DenseCRF2D(FLAGS.max_w / 4, FLAGS.max_h / 4, FLAGS.max_d + 1)
                    # crf.setUnaryEnergy(-unary_energy.reshape(FLAGS.max_d + 1, FLAGS.max_w / 4*FLAGS.max_h / 4))
                    # ref_img_full = np.squeeze((out_images).astype(np.uint8))
                    # ref_img_full=cv2.resize(ref_img_full,(FLAGS.max_w/4,FLAGS.max_h/4))
                    # compat = np.zeros((FLAGS.max_d + 1, FLAGS.max_d + 1), dtype=np.float32)
                    # crf.addPairwiseBilateral(sxy=(80.0, 80.0), srgb=(15.0, 15.0, 15.0),
                    #                          rgbim=ref_img_full, compat=compat, kernel=dcrf.FULL_KERNEL,
                    #                          normalization=dcrf.NORMALIZE_SYMMETRIC)
                    # new_raw = crf.inference(5)
                    # prob_volume = np.array(new_raw).reshape(FLAGS.max_d + 1, FLAGS.max_w / 4, FLAGS.max_h / 4)
                    # prob_index = np.argmax(prob_volume, axis=0)  # h,w
                    # mask = prob_index < (FLAGS.max_d)
                    # out_init_depth_map = mask * (out_start.reshape([]) + prob_index.astype(np.float32) * out_interval)
                except tf.errors.OutOfRangeError:
                    print("all dense finished")  # ==> "End of dataset"
                    break
                duration = time.time() - start_time
                print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (out_index, duration),
                      Notify.ENDC)

                # squeeze output
                out_init_depth_image = np.squeeze(out_init_depth_map)
                print(out_init_depth_image.shape)
                out_prob_map = np.squeeze(out_prob_map)
                out_ref_image = np.squeeze(out_images[0,0])
                out_ref_image = np.squeeze(out_ref_image)
                out_ref_cam = np.squeeze(out_cams)
                out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
                # out_index = np.squeeze(out_index)

                # paths
                init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
                prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
                out_ref_image_path = output_folder + ('/%08d.jpg' % out_index)
                out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)
                out_depth_map_exr_path=output_folder + ('/%08d.exr' % out_index)
                # out_ref_image=cv2.resize(out_ref_image,(out_ref_image.shape[1]/2,out_ref_image.shape[0]/2))
                # out_ref_cam[1,:2,:3]/=2.0
                # #save output
                # out_init_depth_image=cv2.resize(out_init_depth_image,(out_ref_image.shape[1],out_ref_image.shape[0]))
                # out_prob_map = cv2.resize(out_prob_map, (out_ref_image.shape[1], out_ref_image.shape[0]))

                write_pfm(init_depth_map_path, out_init_depth_image)
                #out_init_depth_image[out_prob_map<0.3]=0.0
                imageio.imwrite(out_depth_map_exr_path,out_init_depth_image)
                write_pfm(prob_map_path, out_prob_map)
                out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
                image_file = file_io.FileIO(out_ref_image_path, mode='w')
                scipy.misc.imsave(image_file, out_ref_image)
                write_cam(out_ref_cam_path, out_ref_cam)
                total_step += 1


def main(_):  # pylint: disable=unused-argument
    """ program entrance """
    # generate input path list
    mvs_list = gen_pipeline_mvs_list(FLAGS.dense_folder)
    # mvsnet inference
    mvsnet_pipeline(mvs_list)


if __name__ == '__main__':
    tf.app.run()

#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Loss formulations.
"""

import sys
import math
import tensorflow as tf
import numpy as np

# from crfasrnn.crfrnn_layer import CrfRnnLayer
from loss import mvsnet_regression_loss

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from convgru import ConvGRUCell, BGRUCell
from homography_warping import *

FLAGS = tf.app.flags.FLAGS

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                          tf.global_variables()),
                      tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
          curr_var = name2var[saved_var_name]
          var_shape = curr_var.get_shape().as_list()
          if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """ get probability map from cost volume """

    def _repeat_(x, num_repeats):
        """ repeat each element num_repeats times """
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = tf.shape(cv)[1]

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    # d coordinate (floored and ceiled), batched & flattened
    d_coordinates = tf.reshape((depth_map - depth_start) / depth_interval, [-1])
    d_coordinates_left0 = tf.clip_by_value(tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates_left1 = tf.clip_by_value(d_coordinates_left0 - 1, 0, depth - 1)
    d_coordinates1_right0 = tf.clip_by_value(tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates1_right1 = tf.clip_by_value(d_coordinates1_right0 + 1, 0, depth - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_left1 = tf.stack(
        [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right1 = tf.stack(
        [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)

    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
    prob_map = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map
def  siamese(x):
    x=tf.layers.conv1d(x,16,3,padding='SAME',activation=tf.nn.relu,reuse=tf.AUTO_REUSE,name='s_conv_0')
    x = tf.layers.conv1d(x, 8, 3, padding='SAME', activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='s_conv_1')
    x = tf.layers.conv1d(x, 1, 3, padding='SAME', use_bias=False, reuse=tf.AUTO_REUSE, name='s_conv_2')
    return x
     


def inference_1(images, cams, depth_num, depth_start, depth_end, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """
    # dynamic gpu params
    depth_interval=(depth_end-depth_start)/(depth_num-1)
    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False).get_output()
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True).get_output()
    # ref_tower=[ref_tower.get_output_by_name("sconv2_3"),ref_tower.get_output_by_name("sconv3_3"),
    #            ref_tower.get_output_by_name("sconv4_3")]
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower.get_output())

    # get all homographies
    # scale_view_homographies=[]
    # for scale in range(3):
    #     view_homographies = []
    #     for view in range(1, FLAGS.view_num):
    #         view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
    #         if FLAGS.inverse_depth:
    #             homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
    #                                         depth_start=depth_start, depth_end=depth_end,scale=1.0/(2**scale))
    #         else:
    #             homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
    #                                                       depth_start=depth_start, depth_interval=depth_interval,
    #                                                       scale=1.0 / (2 ** scale))
    #         view_homographies.append(homographies)
    #     scale_view_homographies.append(view_homographies)
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if FLAGS.inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                                      depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval )
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 16]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    # cell1=BGRUCell(8,shape=[FLAGS.max_d],reuse=tf.AUTO_REUSE,name='bgru_1')
    # cell2 = BGRUCell(4, shape=[FLAGS.max_d], reuse=tf.AUTO_REUSE, name='bgru_2')
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=8)
    cell3 = BGRUCell(4, shape=[FLAGS.max_d], reuse=tf.AUTO_REUSE, name='bgru_3')
    cell4 = BGRUCell(4, shape=[FLAGS.max_d], reuse=tf.AUTO_REUSE, name='bgru_4')
    # state0=tf.zeros([FLAGS.batch_size ,FLAGS.max_h/4,FLAGS.max_w/4, 1])
    # conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=16)
    # # cell0 = tf.nn.rnn_cell.GRUCell(num_units=16, reuse=tf.AUTO_REUSE,name='gru_0')
    # cell1 = tf.nn.rnn_cell.GRUCell(num_units=8, reuse=tf.AUTO_REUSE,name='gru_1')
    # cell2 = tf.nn.rnn_cell.GRUCell(num_units=4, reuse=tf.AUTO_REUSE,name='gru_2')
    # state0 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h/4,FLAGS.max_w/4, 16])
    # # state0= tf.zeros([FLAGS.batch_size*feature_shape[1]*feature_shape[2], 16])
    state1 = tf.zeros([FLAGS.batch_size , FLAGS.max_h / 4 , FLAGS.max_w / 4, 8])
    # state2 = tf.zeros([FLAGS.batch_size * FLAGS.max_h / 4 * FLAGS.max_w / 4, 4])
    # state3 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h / 4 ,FLAGS.max_w / 4, 2])
    # prev_prob_image = tf.Variable(tf.zeros(
    #     [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
    #     name='prev_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    # prev_prev_prob_image = tf.Variable(tf.zeros(
    #     [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
    #     name='prev_prev_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    # depth_costs=[]
    # init_map = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])
    # prev_prob_image=tf.assign(prev_prob_image,init_map)
    # prev_prev_prob_image=tf.assign(prev_prev_prob_image,init_map)
    depth_costs = []

    with tf.name_scope('cost_volume_homography'):
        for d in range(depth_num):
            tmp_costs=[]
            # for scale in range(3):
            # compute cost (variation metric)
            ave_feature = ref_tower*1.0
            ave_feature2 = tf.square(ave_feature)
            # ave_feature2=ave_feature-ave_feature
            for view in range(0, FLAGS.view_num - 1):

                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # view_cam = tf.squeeze(tf.slice(cams, [0, view + 1, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
                # homography = get_homography(ref_cam, view_cam, depth=depth,
                #                             scale=1.0 )
                # homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_towers[view], homography)
                # else:
                #     warped_view_feature=view_towers[view][scale]
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
                # ave_feature2=ave_feature2+warped_view_feature-ave_feature
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            # cost=ave_feature2

            cost = ave_feature2 - tf.square(ave_feature)
            # cost=tf.layers.conv2d(cost,16,3,padding='SAME',reuse=tf.AUTO_REUSE,name='test_conv')
            cost,state1=conv_gru1(-cost,state1,scope='conv_gru1')
            depth_costs.append(cost)  # b,h,w,32
            # depths.append(depth)
            # depth = depth + incre_depth_interval
                # cost=tf.concat([cost,ref_tower[scale]],axis=-1)
                # cost=tf.layers.conv2d_transpose(cost,32,3,2**(scale),padding='same',name='upcost_%d'%scale,reuse=tf.AUTO_REUSE)
                # tmp_costs.append(cost)
            # cost=tf.stack(tmp_costs,-1)#b,h,w,32,3
            # cost=tf.reduce_mean(cost,-1)#b,h,w,32
            #b,h,w,32
            # cost=cost+tf.layers.average_pooling2d(cost,pool_size=3,strides=1,padding='same',name='neighbor_cost_pool')
            # cost=tf.cast(cost,dtype=tf.float32)
            # cost=tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,32])

            # ncost=tf.reshape(
            #     tf.extract_image_patches(images=cost, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'),
            #     shape=[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,9,32]
            # )#b,h,w,9,c
            #
            # ncost=tf.reduce_mean((ncost-tf.expand_dims(cost,-2))*2.0/(ncost+tf.expand_dims(cost,-2)),axis=-2)#b,h,w,c
            # cost=cost+0.1*ncost
            # cost=tf.layers.conv2d(-cost,32,3,padding='same',name='cos_conv0',reuse=tf.AUTO_REUSE)
            # cost = UniNetDS2GN({'data': cost}, is_training=True, reuse=tf.AUTO_REUSE)
            # cost, state0 = conv_gru1(cost, state0, scope='conv_gru1')
            # cost = tf.reshape(cost, [-1, 1, 16])
            #
            #cost:b,1,16 state:b,16
            # # cost, state0 = tf.nn.dynamic_rnn(cell=cell0, inputs=-cost, initial_state=state0)
            # cost, state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=cost, initial_state=state1)
            # cost, state2 = tf.nn.dynamic_rnn(cell=cell2, inputs=cost, initial_state=state2)
            # cost=tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,4])
            # cost, state3 = conv_gru1(cost, state3, scope='conv_gru1')
            # cost=tf.concat([cost,resized_image],-1)
            # cost=tf.layers.conv2d(cost,1,3,padding='same',reuse=tf.AUTO_REUSE,name='cos_conv1')#b,h,w,1

            # counts = tf.cast(tf.count_nonzero(tf.atan(cost - 2 * prev_prob_image + prev_prev_prob_image) > 0.7),
            #                  tf.float32)
            # incre_depth_interval = tf.cond(counts <=0.1*FLAGS.max_w*FLAGS.max_h/16,
            #                                lambda: incre_depth_interval * 1.08, lambda: incre_depth_interval)
            #
            # incre_depth_interval = tf.cond(counts > 0.2*FLAGS.max_w*FLAGS.max_h/16 ,
            #                                lambda: depth_interval ,
            #                                lambda: incre_depth_interval)
            # #
            # incre_depth_interval = tf.clip_by_value(incre_depth_interval, depth_interval, depth_interval * 3.0)
            # # tf.keras.backend.print_tensor(incre_depth_interval, message="incre_interval")
            # prev_prev_prob_image = tf.assign(prev_prev_prob_image, prev_prob_image)
            # prev_prob_image = tf.assign(prev_prob_image, cost)
            # # cost = tf.reshape(cost, [-1, 1, 32])  # -1,d,c


            # depth_costs.append(cost)
    cost=tf.stack(depth_costs,-2)#b,h,w,d,32
    with tf.name_scope("rnn_rnn"):
        # prev_row_state_1=tf.zeros([FLAGS.batch_size,FLAGS.max_w/4,FLAGS.max_d,8])#b,w,d,c
        # prev_row_state_2 = tf.zeros([FLAGS.batch_size, FLAGS.max_w / 4, FLAGS.max_d, 4])  # b,w,d,c
        state3 = tf.zeros([FLAGS.batch_size*FLAGS.max_h/4, FLAGS.max_d, 4])  # b,w,d,c
        cost3,state3=tf.nn.dynamic_rnn(cell3,inputs=tf.reshape(cost,[-1,FLAGS.max_w/4,FLAGS.max_d,8]),initial_state=state3)
        state4 = tf.zeros([FLAGS.batch_size*FLAGS.max_w/4, FLAGS.max_d, 4])  # b,w,d,c
        cost4, state4 = tf.nn.dynamic_rnn(cell4, inputs=tf.reshape(tf.transpose(cost,[0,2,1,3,4]), [-1, FLAGS.max_h / 4, FLAGS.max_d, 8]),
                                          initial_state=state4)
        cost3=tf.reshape(cost3,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,FLAGS.max_d,4])
        cost4 = tf.reshape(cost4, [FLAGS.batch_size, FLAGS.max_w / 4, FLAGS.max_h/ 4, FLAGS.max_d,4])
        cost4=tf.transpose(cost4,[0,2,1,3,4])
        cost=tf.concat([cost3,cost4],-1)#b,h,w,d,c3
        cost=tf.keras.layers.SpatialDropout3D(0.5)(cost)
        cost=tf.layers.conv3d(cost,1,3,1,padding='SAME',reuse=tf.AUTO_REUSE,name='final_conv')



    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        depth_costs = tf.transpose(tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,FLAGS.max_d,1]), [0, 3, 1, 2, 4])  # b,h,w,d,1 -> b,d,h,w,1
        probability_volume = tf.nn.softmax(depth_costs, axis=1, name='prob_volume')  # b,d,h,w,1

        return probability_volume
  
def inference_2(images, cams, depth_num, depth_start, depth_end, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """
    # dynamic gpu params
    depth_interval=(depth_end-depth_start)/(depth_num-1)
    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_tower=ref_tower.get_output()
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
        view_towers.append(view_tower.get_output())
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    # build cost volume by differentialble homography
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=16)
    # cell0 = tf.nn.rnn_cell.GRUCell(num_units=16, reuse=tf.AUTO_REUSE,name='gru_0')
    cell1 = tf.nn.rnn_cell.GRUCell(num_units=8, reuse=tf.AUTO_REUSE,name='gru_1')
    cell2 = tf.nn.rnn_cell.GRUCell(num_units=4, reuse=tf.AUTO_REUSE,name='gru_2')
    state0 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h/4,FLAGS.max_w/4, 16])
    # state0= tf.zeros([FLAGS.batch_size*feature_shape[1]*feature_shape[2], 16])
    row_state1 = tf.zeros([FLAGS.batch_size * FLAGS.max_h / 4 , 8])
    row_state2 = tf.zeros([FLAGS.batch_size * FLAGS.max_h / 4 ,  4])
    col_state1 = tf.zeros([FLAGS.batch_size * FLAGS.max_w / 4 , 8])
    col_state2 = tf.zeros([FLAGS.batch_size * FLAGS.max_w / 4 , 4])
    # state3 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h / 4 ,FLAGS.max_w / 4, 2])

    # depth_costs=[]

    depth_costs = []
    with tf.name_scope('cost_volume_homography'):
        for d in range(depth_num):
            tmp_costs=[]
            # for scale in range(3):
            # compute cost (variation metric)
            ave_feature = ref_tower*1.0
            ave_feature2 = tf.square(ave_feature)
            # ave_feature2=ave_feature-ave_feature
            for view in range(0, FLAGS.view_num - 1):

                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_towers[view], homography)
                # else:
                #     warped_view_feature=view_towers[view][scale]
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
                # ave_feature2=ave_feature2+warped_view_feature-ave_feature
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            # cost=ave_feature2
            cost = ave_feature2 - tf.square(ave_feature)
            # cost=tf.concat([cost,ref_tower[scale]],axis=-1)
            cost=tf.layers.conv2d_transpose(cost,32,3,1,padding='same',name='upcost_%d'%0,reuse=tf.AUTO_REUSE)
            cost=tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,32])
            cost, state0 = conv_gru1(cost, state0, scope='conv_gru1')#b,h,w,c
            row_cost = tf.reshape(cost, [-1, FLAGS.max_w/4, 16])
            row_cost, row_state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=row_cost, initial_state=row_state1)
            row_cost, row_state2 = tf.nn.dynamic_rnn(cell=cell2, inputs=row_cost, initial_state=row_state2)
            row_cost=tf.reshape(row_cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,4])
            col_cost=tf.reshape(tf.transpose(cost,[0,2,1,3]),[-1,FLAGS.max_h/4,16])
            col_cost, col_state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=col_cost, initial_state=col_state1)
            col_cost, col_state2 = tf.nn.dynamic_rnn(cell=cell2, inputs=col_cost, initial_state=col_state2)
            col_cost=tf.transpose(tf.reshape(col_cost,[FLAGS.batch_size,FLAGS.max_w/4,FLAGS.max_h/4,4]),[0,2,1,3])
            cost=tf.concat([col_cost,row_cost],-1)#b,h,w,8
        
           
            cost=tf.layers.conv2d(cost,1,3,padding='same',reuse=tf.AUTO_REUSE,name='cos_conv1')#b,h,w,1
            depth_costs.append(cost)
       

    with tf.name_scope('soft_arg_min'):
      
        filtered_cost_volume=tf.stack(depth_costs,1)#b,d,h,w,1
        probability_volume = tf.nn.softmax(filtered_cost_volume, axis=1, name='prob_volume')  # b,d,h,w,1

        return probability_volume

def inference(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """
    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    else:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=tf.AUTO_REUSE)
    ref_tower=[ref_tower.get_output_by_name("sconv2_3"),ref_tower.get_output_by_name("sconv3_3"),
               ref_tower.get_output_by_name("sconv4_3")]
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = SNetDS2GN({'data': view_image}, is_training=True, reuse=tf.AUTO_REUSE)
        view_towers.append([view_tower.get_output_by_name("sconv2_3"),view_tower.get_output_by_name("sconv3_3"),view_tower.get_output_by_name("sconv4_3")])

    # get all homographies
    scale_view_homographies=[]
    for scale in range(3):
        view_homographies = []
        for view in range(1, FLAGS.view_num):
            view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
            if FLAGS.inverse_depth:
                homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_end=depth_end,scale=1.0/(2**scale))
            else:
                homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                                          depth_start=depth_start, depth_interval=depth_interval,
                                                          scale=1.0 / (2 ** scale))
            view_homographies.append(homographies)
        scale_view_homographies.append(view_homographies)


    # build cost volume by differentialble homography
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=16)
    # cell0 = tf.nn.rnn_cell.GRUCell(num_units=16, reuse=tf.AUTO_REUSE,name='gru_0')
    cell1 = tf.nn.rnn_cell.GRUCell(num_units=8, reuse=tf.AUTO_REUSE,name='gru_1')
    cell2 = tf.nn.rnn_cell.GRUCell(num_units=4, reuse=tf.AUTO_REUSE,name='gru_2')
    state0 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h/4,FLAGS.max_w/4, 16])
    # state0= tf.zeros([FLAGS.batch_size*feature_shape[1]*feature_shape[2], 16])
    state1 = tf.zeros([FLAGS.batch_size * FLAGS.max_h / 4 * FLAGS.max_w / 4, 8])
    state2 = tf.zeros([FLAGS.batch_size * FLAGS.max_h / 4 * FLAGS.max_w / 4, 4])
    # state3 = tf.zeros([FLAGS.batch_size ,FLAGS.max_h / 4 ,FLAGS.max_w / 4, 2])

    # depth_costs=[]

    depth_costs = []
    with tf.name_scope('cost_volume_homography'):
        for d in range(depth_num):
            tmp_costs=[]
            for scale in range(3):
                # compute cost (variation metric)
                ave_feature = ref_tower[scale]*1.0
                ave_feature2 = tf.square(ave_feature)
                # ave_feature2=ave_feature-ave_feature
                for view in range(0, FLAGS.view_num - 1):

                    homography = tf.slice(scale_view_homographies[scale][view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                    homography = tf.squeeze(homography, axis=1)
                    # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                    warped_view_feature = tf_transform_homography(view_towers[view][scale], homography)
                    # else:
                    #     warped_view_feature=view_towers[view][scale]
                    ave_feature = ave_feature + warped_view_feature
                    ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
                    # ave_feature2=ave_feature2+warped_view_feature-ave_feature
                ave_feature = ave_feature / FLAGS.view_num
                ave_feature2 = ave_feature2 / FLAGS.view_num
                # cost=ave_feature2
                cost = ave_feature2 - tf.square(ave_feature)
                # cost=tf.concat([cost,ref_tower[scale]],axis=-1)
                cost=tf.layers.conv2d_transpose(cost,32,3,2**(scale),padding='same',name='upcost_%d'%scale,reuse=tf.AUTO_REUSE)
                tmp_costs.append(cost)
            cost=tf.stack(tmp_costs,-1)#b,h,w,32,3
            cost=tf.reduce_mean(cost,-1)#b,h,w,32
            #b,h,w,32
            # cost=cost+tf.layers.average_pooling2d(cost,pool_size=3,strides=1,padding='same',name='neighbor_cost_pool')
            # cost=tf.cast(cost,dtype=tf.float32)
            cost=tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,32])
            cost, state0 = conv_gru1(cost, state0, scope='conv_gru1')
            cost = tf.reshape(cost, [-1, 1, 16])

            # cost, state0 = tf.nn.dynamic_rnn(cell=cell0, inputs=-cost, initial_state=state0)
            cost, state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=cost, initial_state=state1)
            cost, state2 = tf.nn.dynamic_rnn(cell=cell2, inputs=cost, initial_state=state2)
            cost=tf.reshape(cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,4])
            # cost, state3 = conv_gru1(cost, state3, scope='conv_gru1')
            # cost=tf.concat([cost,resized_image],-1)
            cost=tf.layers.conv2d(cost,1,3,padding='same',reuse=tf.AUTO_REUSE,name='cos_conv1')#b,h,w,1
            depth_costs.append(cost)
            # cost = tf.reshape(cost, [-1, 1, 32])  # -1,d,c


            # depth_costs.append(cost)

    with tf.name_scope('soft_arg_min'):
        filtered_cost_volume=tf.stack(depth_costs,1)#b,d,h,w,1
        probability_volume = tf.nn.softmax(filtered_cost_volume, axis=1, name='prob_volume')  # b,d,h,w,1

        return probability_volume
   

def inference_mem(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_features[view], homography)
                warped_view_feature = tf_transform_homography(view_features[view], homography)
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    # return filtered_depth_map,
    return estimated_depth_map, prob_map

def get_homo_matrix(cams,depth_num,depth_start,depth_interval,is_master_gpu):
    # get all homographies
    view_homographies = []
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    return view_homographies

def get_features(images,is_master_gpu=True):
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    # image feature extraction

    ref_tower = UniNetDS2({'data': ref_image}, is_training=True, reuse=is_master_gpu==False).get_output()

    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS2({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower.get_output())
    return ref_tower ,view_towers

def get_prob_recurrent(ref_features,view_features,homos,depth_num,step):
    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)
    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth_costs = []
        for d in range(0,depth_num,step):

            # compute cost (variation metric)
            ave_feature = ref_features
            ave_feature2 = tf.square(ref_features)

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    homos[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_features[view], homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost=ave_feature2 - tf.square(ave_feature)  # = ave_feature2 - tf.square(ave_feature)#b,h,w,c


            reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
            reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
            reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
            reg_cost = tf.layers.conv2d(
                reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            depth_costs.append(reg_cost)

        prob_volume = tf.stack(depth_costs, axis=1)  # b,d,h,w,1
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')

        return prob_volume

def inference_prob_recurrent(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer disparity image from stereo images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = SNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        # view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        # homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
        #                                 depth_start=depth_start, depth_interval=depth_interval)
        # view_homographies.append(homographies)
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if FLAGS.inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h/4, FLAGS.max_w/4, 16]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)
    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth=depth_start
        costs=[]
      
        for d in range(depth_num):
          
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # view_cam = tf.squeeze(tf.slice(cams, [0, view+1, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
                # homography = get_homography(ref_cam, view_cam, depth=depth,
                #                             scale=1.0 )
                # homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost=ave_feature2 - tf.square(ave_feature) #= ave_feature2 - tf.square(ave_feature)
            # gru
            reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
            reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
            reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
            reg_cost = tf.layers.conv2d(
                reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            costs.append(reg_cost)

        prob_volume = tf.stack(costs, axis=1)
       
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')
       
    return prob_volume

def inference_winner_take_all(images, cams, depth_num, depth_start, depth_end, 
                              is_master_gpu=True, reg_type='GRU', inverse_depth=False):
    """ infer disparity image from stereo images and cameras """

    if not inverse_depth:
        depth_interval = (depth_end - depth_start) / (tf.cast(depth_num, tf.float32) - 1)

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = SNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = SNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # gru unit
    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h, FLAGS.max_w, 16]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)

    # initialize variables
    exp_sum = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='exp_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    depth_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='depth_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    max_prob_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='max_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    init_map = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])

    # define winner take all loop
    def body(depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre):
        """Loop body."""

        # calculate cost 
        ave_feature = ref_tower.get_output()
        ave_feature2 = tf.square(ref_tower.get_output())
        for view in range(0, FLAGS.view_num - 1):
            homographies = view_homographies[view]
            homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
            homography = homographies[depth_index]
            # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
            warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
            ave_feature = ave_feature + warped_view_feature
            ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
        ave_feature = ave_feature / FLAGS.view_num
        ave_feature2 = ave_feature2 / FLAGS.view_num
        cost = ave_feature2 - tf.square(ave_feature)
        cost.set_shape([FLAGS.batch_size, feature_shape[1], feature_shape[2], 16])

        # gru
        reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
        reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
        reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
        reg_cost = tf.layers.conv2d(
            reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
        prob = tf.exp(reg_cost)

        # index
        d_idx = tf.cast(depth_index, tf.float32) 
        if inverse_depth:
            inv_depth_start = tf.div(1.0, depth_start)
            inv_depth_end = tf.div(1.0, depth_end)
            inv_interval = (inv_depth_start - inv_depth_end) / (tf.cast(depth_num, 'float32') - 1)
            inv_depth = inv_depth_start - d_idx * inv_interval
            depth = tf.div(1.0, inv_depth)
        else:
            depth = depth_start + d_idx * depth_interval
        temp_depth_image = tf.reshape(depth, [FLAGS.batch_size, 1, 1, 1])
        temp_depth_image = tf.tile(
            temp_depth_image, [1, feature_shape[1], feature_shape[2], 1])

        # update the best
        update_flag_image = tf.cast(tf.less(max_prob_image, prob), dtype='float32')
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
        new_depth_image = update_flag_image * temp_depth_image + (1 - update_flag_image) * depth_image
        max_prob_image = tf.assign(max_prob_image, new_max_prob_image)
        depth_image = tf.assign(depth_image, new_depth_image)

        # update counter
        exp_sum = tf.assign_add(exp_sum, prob)
        depth_index = tf.add(depth_index, incre)

        return depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre
    
    # run forward loop
    exp_sum = tf.assign(exp_sum, init_map)
    depth_image = tf.assign(depth_image, init_map)
    max_prob_image = tf.assign(max_prob_image, init_map)
    depth_index = tf.constant(0)
    incre = tf.constant(1)
    cond = lambda depth_index, *_: tf.less(depth_index, depth_num)
    _, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre = tf.while_loop(
        cond, body
        , [depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre]
        , back_prop=False, parallel_iterations=1)

    # get output
    forward_exp_sum = exp_sum + 1e-7
    forward_depth_map = depth_image
    return forward_depth_map, max_prob_image / forward_exp_sum

def depth_refine(init_depth_map, image, depth_num, depth_start, depth_interval, is_master_gpu=True):

    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.div(init_depth_map - depth_start_mat, depth_scale_mat)

    # resize normalized image to the same size of depth image
    resized_image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])

    # refinement network
    if is_master_gpu:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=False)
    else:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                        is_training=True, reuse=True)
    norm_depth_map = norm_depth_tower.get_output()

    # denormalize depth map
    refined_depth_map = tf.multiply(norm_depth_map, depth_scale_mat) + depth_start_mat

    return refined_depth_map


    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)#B,H,W,D

    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.div(init_depth_map - depth_start_mat, depth_scale_mat)

    # resize normalized image to the same size of depth image
    resized_image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])

    # refinement network
    if is_master_gpu:
        norm_depth_tower = RefineNet_1({'color_image': resized_image, 'depth_images': init_norm_depth_map},
                                        is_training=True, reuse=False)
    else:
        norm_depth_tower = RefineNet_1({'color_image': resized_image, 'depth_images': init_norm_depth_map},
                                        is_training=True, reuse=True)
    norm_depth_map = norm_depth_tower.get_output()
    # denormalize depth map
    refined_depth_map = tf.multiply(norm_depth_map, depth_scale_mat) + depth_start_mat

    return refined_depth_map


def generate_normalmap(depth_map,ref_cam):
    return None


def depth_normal_refine(images,cams,depth_map,normal_map):

    return None



def inference_prob_recurrent_1(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer disparity image from stereo images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = SNetDS2GN_1({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = SNetDS2GN_1({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = SNetDS2GN_1({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        # view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        # homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
        #                                 depth_start=depth_start, depth_interval=depth_interval)
        # view_homographies.append(homographies)
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if FLAGS.inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 4
    feature_shape = [FLAGS.batch_size, FLAGS.max_h/4, FLAGS.max_w/4, 16]
    gru_input_shape = [feature_shape[1], feature_shape[2]]  
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)

    cell0 = tf.nn.rnn_cell.GRUCell(num_units=8, reuse=tf.AUTO_REUSE,name='gru_1')
    cell1 = tf.nn.rnn_cell.GRUCell(num_units=4, reuse=tf.AUTO_REUSE,name='gru_2')
   
    state0 = tf.zeros([FLAGS.batch_size * FLAGS.max_h/4 * FLAGS.max_w/4, 8])
    state1 = tf.zeros([FLAGS.batch_size * FLAGS.max_h/4 * FLAGS.max_w/4 , 4])

    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth=depth_start
        costs=[]
      
        for d in range(depth_num):
          
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # view_cam = tf.squeeze(tf.slice(cams, [0, view+1, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
                # homography = get_homography(ref_cam, view_cam, depth=depth,
                #                             scale=1.0 )
                # homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost=  tf.square(ave_feature) -ave_feature2#= ave_feature2 - tf.square(ave_feature)
            # gru
            # reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
            # reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
            _1d_cost = tf.reshape(cost, [-1, 1, 16])
            # cost, state0 = tf.nn.dynamic_rnn(cell=cell0, inputs=-cost, initial_state=state0)
            _1d_cost, state0 = tf.nn.dynamic_rnn(cell=cell0, inputs=_1d_cost, initial_state=state0)
            _1d_cost, state1 = tf.nn.dynamic_rnn(cell=cell1, inputs=_1d_cost, initial_state=state1)
            _1d_cost=tf.reshape(_1d_cost,[FLAGS.batch_size,FLAGS.max_h/4,FLAGS.max_w/4,4])
            _2d_cost, state3 = conv_gru3(cost, state3, scope='conv_gru3')
            
            reg_cost=tf.concat([_1d_cost,_2d_cost],axis=-1)
            reg_cost = tf.layers.conv2d(
                reg_cost, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            costs.append(reg_cost)

        prob_volume = tf.stack(costs, axis=1)
       
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')
       
    return prob_volume


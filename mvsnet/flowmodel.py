import tensorflow as tf
import sys
sys.path.append("../")
from cnn_wrapper.mvsnet import *
from convgru import ConvGRUCell, BGRUCell
from homography_warping import *
FLAGS = tf.app.flags.FLAGS
import numpy as np

def  update_cams(cams,scale=1):
    scales=tf.tile(tf.reshape(tf.constant([scale,scale,1,1],tf.float32),[4,1]),[1,4])
    ones=tf.ones((4,4),dtype=tf.float32)
    scales=scales*ones
    scales=tf.reshape(tf.concat([ones,scales],0),[1,1,2,4,4])
    scale_cams=scales*cams
    return scale_cams

def flow_pipline(ref_feature,view_features,cams,flow,radius,shape,index=0):
    
    batch_size=shape[0]    
    height=shape[1]
    width=shape[2]
    channel=shape[3]   
    conv_gru0 = ConvGRUCell(shape=[None, None], kernel=[3, 3], filters=channel)
    conv_gru1 = ConvGRUCell(shape=[None, None], kernel=[3, 3], filters=channel/2)
    conv_gru2 = ConvGRUCell(shape=[None, None], kernel=[3, 3], filters=channel/4)
    state0 = tf.zeros([FLAGS.batch_size, height, width, channel])
    state1 = tf.zeros([FLAGS.batch_size, height, width, channel/2])
    state2 = tf.zeros([FLAGS.batch_size, height, width, channel/4])
    costs=[]
    flow_dists=[]
    ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
    view_cam=tf.squeeze(tf.slice(cams,[0,1,0,0,0],[-1,1,-1,-1,-1]),1)
    p,q=PQ(ref_cam,view_cam,shape)
    x_linspace = tf.linspace(0., tf.cast(width, 'float32'), width)
    y_linspace = tf.linspace(0., tf.cast(height, 'float32'), height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates=tf.tile(tf.reshape(x_coordinates,[1,height,width,1]),[batch_size,1,1,1])
   
    depths=[]

    for rx in np.arange(-width/8.0,width/8.0,width/4.0/radius):
        # xflow=flow+float(rx)
        xflow=flow+float(rx)
        coords=x_coordinates+xflow
        depth=grad_d(p,q,coords)
        depths.append(depth)
        flow_dists.append(tf.reshape(xflow,[batch_size,height,width,1]))
      
        ave_feature=ref_feature*1.0
        ave_feature2=tf.square(ref_feature)

        for view in range(0,FLAGS.view_num-1):
            view_cam=tf.squeeze(tf.slice(cams,[0,1+view,0,0,0],[-1,1,-1,-1,-1]),1)
            view_feature=tf.squeeze(tf.slice(view_features,[0,view,0,0,0],[-1,1,-1,-1,-1]),1)
            warped,_,_=reprojection_depth(view_feature,ref_cam,view_cam,depth)
            ave_feature+=warped
            ave_feature2+=tf.square(warped)
        ave_feature = ave_feature / FLAGS.view_num
        ave_feature2 = ave_feature2 / FLAGS.view_num
        cost=  ave_feature2-tf.square(ave_feature) #DX=E(X^2)-(EX)^2
        """
        rnn model
        """

        cost,state0=conv_gru0(-cost,state0,scope='conv_gru_0')
        cost,state1=conv_gru1(cost,state1,scope='conv_gru_1')
        cost,state2=conv_gru2(cost,state2,scope='conv_gru_2')
       
        cost=tf.layers.conv2d(cost,1,3,1,'SAME',reuse=tf.AUTO_REUSE,name='cost_conv')

        cost=tf.reshape(cost,[batch_size,height,width,1])
        costs.append(cost)
    costs=tf.stack(costs,1)#b,d,h,w,1
    costs=tf.nn.softmax(costs,1)
    flow_dists=tf.stack(flow_dists,1)#b,d,h,w,1
    flow0=tf.reduce_sum((flow_dists*costs),1)#b,h,w,1
    coords=x_coordinates+ flow0
    depth=grad_d(p,q,coords)
    depths=tf.stack(depths,1)
    depth_min=tf.reduce_min(depths,1)
    depth_max=tf.reduce_max(depths,1)
    return flow0,depth,depth_min,depth_max

def conv(inputs,filter,kernel_size,stride,padding,name,reuse):
    return  tf.layers.conv2d(inputs,filter,kernel_size,stride,padding,name=name,reuse=reuse)   
def conv_bn(inputs,filter,kernel_size,stride,padding,name,reuse):
    x=conv(inputs,filter,kernel_size,stride,padding,name=name,reuse=reuse)
    x=tf.contrib.layers.batch_norm(x)
    return tf.nn.relu(x)
def deconv_bn(inputs,filter,kernel_size,stride,padding,name,reuse):
    x=tf.layers.conv2d_transpose(inputs,filter,kernel_size,stride,padding,name=name,reuse=reuse)
    x=tf.contrib.layers.batch_norm(x)
    return tf.nn.relu(x)


def depth_inference(images,cams):
    #step 1 : feature extractor
    with tf.name_scope("feature_model"):
        #unet model
        #encode 
        batch_size=FLAGS.batch_size
        height=FLAGS.max_h
        width=FLAGS.max_w
       
        images=tf.reshape(images,[-1,height,width,3])
        conv0_0=conv_bn(images,16,1,1,'SAME',name='conv0_0',reuse=tf.AUTO_REUSE)
        conv0_1=conv_bn(conv0_0,16,1,1,'SAME',name='conv0_1',reuse=tf.AUTO_REUSE)
        conv0_2=conv_bn(conv0_1,16,1,1,'SAME',name='conv0_2',reuse=tf.AUTO_REUSE) #f0

        conv1_0=conv_bn(conv0_2,32,3,2,'SAME',name='conv1_0',reuse=tf.AUTO_REUSE)
        conv1_1=conv_bn(conv1_0,32,3,1,'SAME',name='conv1_1',reuse=tf.AUTO_REUSE)
        conv1_2=conv_bn(conv1_1,32,3,1,'SAME',name='conv1_2',reuse=tf.AUTO_REUSE)  #f1

        conv2_0=conv_bn(conv1_2,64,3,2,'SAME',name='conv2_0',reuse=tf.AUTO_REUSE)
        conv2_1=conv_bn(conv2_0,64,3,1,'SAME',name='conv2_1',reuse=tf.AUTO_REUSE)
        conv2_2=conv(conv2_1,32,3,1,'SAME',name='conv2_2',reuse=tf.AUTO_REUSE) #f2
        #decode 
        
        #end of unet model 
       
        
        channel=tf.shape(conv2_2)[3]
        features=tf.reshape(conv2_2,[FLAGS.batch_size,FLAGS.view_num,height/4,width/4,32])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        radius=[16,32,64]
        flow=tf.zeros([batch_size,height/4,width/4,1])
        cams=update_cams(cams,0.25)
        flow,depth0,_,_=flow_pipline(ref_feature,view_features,cams,flow,radius[0],[batch_size,height/4,width/4,32],0)
        conv2_2=tf.nn.relu(conv2_2)
        dconv3_0=deconv_bn(conv2_2,64,3,2,'SAME',reuse=tf.AUTO_REUSE,name='dconv3_0')
        conv3_1=conv_bn(tf.concat([dconv3_0,conv1_2],-1),32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_1')
        conv3_2=conv_bn(conv3_1,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_2')
        conv3_3=conv(conv3_2,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_3')
        features=tf.reshape(conv3_3,[FLAGS.batch_size,FLAGS.view_num,height/2,width/2,32])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        cams=update_cams(cams,2)
        up_flow=tf.image.resize_images(flow,(height/2,width/2))*2.0

        flow,depth1,_,_=flow_pipline(ref_feature,view_features,cams,up_flow,radius[1],[batch_size,height/2,width/2,32],1)
      
        conv3_3=tf.nn.relu(conv3_3)
        dconv4_0=deconv_bn(conv3_3,32,1,2,'SAME',reuse=tf.AUTO_REUSE,name='dconv4_0')
        conv4_1=conv_bn(tf.concat([dconv4_0,conv0_2],-1),32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_1')
        conv4_2=conv_bn(conv4_1,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_2')
        conv4_3=conv(conv4_2,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_3')
        features=tf.reshape(conv4_3,[FLAGS.batch_size,FLAGS.view_num,height,width,32])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        cams=update_cams(cams,2)
        # up_flow=tf.image.resize_images(flow,(height,width))*2.0
        up_flow=tf.zeros([batch_size,height,width,1])
        flow,depth2,depth_min,depth_max=flow_pipline(ref_feature,view_features,cams,up_flow,radius[2],[batch_size,height,width,32],2)
       
        return depth2,depth1,depth0
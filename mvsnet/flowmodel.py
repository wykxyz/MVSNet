import tensorflow as tf
import sys
sys.path.append("../")
from cnn_wrapper.mvsnet import *
from convgru import ConvGRUCell, BGRUCell
from homography_warping import *
FLAGS = tf.app.flags.FLAGS
def flow2depth(flow,ref_cam,view_cam):
    
    shape=tf.shape(flow)
    batch_size=shape[0]
    height=shape[1]
    width=shape[2]
    ref_r=tf.squeeze(tf.slice(ref_cam,[0,0,0,0],[-1,1,3,3]),1)#b,3,3
    ref_t=tf.squeeze(tf.slice(ref_cam,[0,0,0,3],[-1,1,3,1]),1)#b,3,1
    ref_k=tf.squeeze(tf.slice(ref_cam,[0,1,0,0],[-1,1,3,3]),1)#b,3,3
    ref_k_inv=tf.matrix_inverse(ref_k)
    ref_r_inv=tf.transpose(ref_r,[0,2,1])
    view_r=tf.squeeze(tf.slice(view_cam,[0,0,0,0],[-1,1,3,3]),1)#b,3,3
    view_t=tf.squeeze(tf.slice(view_cam,[0,0,0,3],[-1,1,3,1]),1)#b,3,1
    view_k=tf.squeeze(tf.slice(view_cam,[0,1,0,0],[-1,1,3,3]),1)#b,3,3
    ref_c=-tf.matmul(tf.transpose(ref_r,[0,2,1]),ref_t)#b,3,1
    view_c=-tf.matmul(tf.transpose(view_r,[0,2,1]),view_t)#b,3,1
    del_c=ref_c-view_c
    x_linspace = tf.linspace(0.0, tf.cast(width, 'float32') , width)
    y_linspace = tf.linspace(0.0, tf.cast(height, 'float32'), height)
    ori_coord = tf.meshgrid(x_linspace, y_linspace)
    ones=tf.ones([batch_size,height,width,1])
    
    coords=tf.expand_dims(tf.stack(ori_coord,-1),0)#1,h,w,2
    coords=tf.tile(coords,[batch_size,1,1,1])
    view_coords=coords+flow
    coords=tf.concat([coords,ones],-1)
    view_coords=tf.concat([view_coords,ones],-1)

    coords=tf.transpose(coords,[0,3,1,2])#b,3,h,w
    # view_coords=tf.transpose(view_coords,[0,3,1,2])
    coords=tf.reshape(coords,[batch_size,3,-1])
    # view_coords=tf.rshape(view_coords,[batch_size,3,-1])
    coords=tf.matmul(tf.matmul(ref_r_inv,ref_k_inv),coords)
    coords=tf.reshape(tf.transpose(coords,[0,2,1]),[batch_size,height,width,3])
    del_c=tf.tile(tf.reshape(del_c,[batch_size,1,1,3]),[1,height,width,1])#b,h,w,3
    a=tf.cross(coords,view_coords)
    b=tf.cross(del_c,view_coords)
    mask=tf.cast(tf.equal(a,0.0),tf.float32)
    a=a+mask*1e-7
    depth=tf.reduce_mean(tf.divide(-b,a),-1,keepdims=True)
    return depth

def  update_cams(cams,scale=1):
    scales=tf.tile(tf.reshape(tf.constant([scale,scale,1,1],tf.float32),[4,1]),[1,4])
    ones=tf.ones((4,4),dtype=tf.float32)
    scales=scales*ones
    scales=tf.reshape(tf.concat([ones,scales],0),[1,1,2,4,4])
    scale_cams=scales*cams
    return scale_cams

def flowing(inputs,height,width,bias_x,bias_y):

    x_linspace = tf.linspace(float(bias_x), tf.cast(width, 'float32')+float(bias_x) , width)
    y_linspace = tf.linspace(float(bias_y), tf.cast(height, 'float32')+float(bias_y), height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates=tf.reshape(x_coordinates,[1,-1])
    y_coordinates=tf.reshape(y_coordinates,[1,-1])
    x_coordinates = tf.expand_dims(x_coordinates, 0)
    y_coordinates = tf.expand_dims(y_coordinates, 0)
    x_coordinates=tf.reshape(x_coordinates,[-1])
    y_coordinates=tf.reshape(y_coordinates,[-1])
    warped=interpolate(inputs,x_coordinates,y_coordinates)
    flow_dist=tf.stack([x_coordinates,y_coordinates],-1)#bn,2
    flow_dist=tf.reshape(flow_dist,[1,height,width,2])
    return warped,flow_dist
def concatflow(flow0,flow1):
    shape=tf.shape(flow0)
    batch_size=shape[0]
    height=shape[1]
    width=shape[2]
    x_linspace = tf.linspace(0.0, tf.cast(width, 'float32') , width)
    y_linspace = tf.linspace(0.0, tf.cast(height, 'float32'), height)
    ori_coord = tf.meshgrid(x_linspace, y_linspace)
    coords=tf.expand_dims(tf.stack(ori_coord,-1),0)#1,h,w,2
    coords=tf.tile(coords,[batch_size,1,1,1])
    view_coords=coords+flow0
    view_coords=tf.reshape(view_coords,[-1,2])
    x_coords=tf.squeeze(tf.slice(view_coords,[0,0],[-1,1]),-1)
    y_coords=tf.squeeze(tf.slice(view_coords,[0,1],[-1,1]),-1)
    dist_flow=interpolate(flow1,x_coords,y_coords)
    return flow0+dist_flow
def depthflow(p,q,x,height,width):
    x_linspace = tf.linspace(0.0, tf.cast(width, 'float32') , width)
    y_linspace = tf.linspace(0.0, tf.cast(height, 'float32'), height)
    x_coord,y_coord = tf.meshgrid(x_linspace, y_linspace)
    x_coord=x_coord+x
    depth=grad_d(p,q,x_coord)

def flow_pipline(ref_feature,view_features,cams,flow,radius,conv_gru,state,channel,index=0):
    shape=tf.shape(ref_feature)
    batch_size=shape[0]
    
    height=shape[1]
    width=shape[2]
    # channel=shape[3]
    
    costs=[]
    flow_dists=[]
    ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
    view_cam=tf.squeeze(tf.slice(cams,[0,1,0,0,0],[-1,1,-1,-1,-1]),1)
    p,q=PQ(ref_cam,view_cam,shape)

    for rx in range(-(radius/2),radius/2,1):
        xflow=flow+rx
        depth=grad_d(p,q,xflow)
        flow_dists.append(tf.reshape(xflow,[batch_size,height,width,1]))
        # ref_cam=tf.squeeze(tf.slice(cams,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        # view_cam=tf.squeeze(tf.slice(cams,[0,1,0,0,0],[-1,1,-1,-1,-1]),1)
        ave_feature=ref_feature
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
    
        cost=tf.reshape(cost,[FLAGS.batch_size,height,width,channel])
        cost,state=conv_gru(-cost,state,scope='conv_gru_%d'%index)
        cost=tf.layers.conv2d(cost,1,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv_%d'%index)
        costs.append(cost)
    costs=tf.stack(costs,1)#b,d,h,w,1
    costs=tf.nn.softmax(costs,1)
    flow_dists=tf.stack(flow_dists,1)#b,d,h,w,1
    flow0=tf.reduce_sum((flow_dists*costs),1)#b,h,w,1
    flow0=tf.reshape(flow0,[batch_size,height,width,1])
    # flow=concatflow(flow,flow0)
    # depth=flow2depth(flow0,ref_cam,view_cam)
    depth=grad_d(p,q,flow0)
    return flow0,depth
def flow_warping(inputs,flow):
    flow=tf.reshape(flow,[-1,2])
    shape=tf.shape(inputs)
    height=shape[1]
    width=shape[2]
    x_linspace = tf.linspace(0.5, tf.cast(width, 'float32')+0.5 , width)
    y_linspace = tf.linspace(0.5, tf.cast(height, 'float32')+0.5, height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates=tf.reshape(x_coordinates,[-1,1])
    y_coordinates=tf.reshape(y_coordinates,[-1,1])
    # x_coordinates = tf.expand_dims(x_coordinates, 0)
    # y_coordinates = tf.expand_dims(y_coordinates, 0)
    # x_coordinates=tf.reshape(x_coordinates,[-1])
    # y_coordinates=tf.reshape(y_coordinates,[-1])
    x_coordinates+=tf.slice(flow,[0,0],[-1,1])
    y_coordinates+=tf.slice(flow,[0,1],[-1,1])
    x_coordinates=tf.reshape(x_coordinates,[-1])
    y_coordinates=tf.reshape(y_coordinates,[-1])
    
    warped=interpolate(inputs,x_coordinates,y_coordinates)
    warped=tf.reshape(warped,tf.shape(inputs))
    return warped

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
        conv2_2=conv(conv2_1,64,3,1,'SAME',name='conv2_2',reuse=tf.AUTO_REUSE) #f2
        #decode 
        
        #end of unet model 
        conv_gru0 = ConvGRUCell(shape=[height/4, width/4], kernel=[3, 3], filters=32)
        conv_gru1 = ConvGRUCell(shape=[height/2, width/2], kernel=[3, 3], filters=16)
        conv_gru2 = ConvGRUCell(shape=[height, width], kernel=[3, 3], filters=8)
        state0 = tf.zeros([FLAGS.batch_size, height/4, width/4, 32])
        state1 = tf.zeros([FLAGS.batch_size, height/2, width/2, 16])
        state2 = tf.zeros([FLAGS.batch_size, height, width, 8])
        
        channel=tf.shape(conv2_2)[3]
        features=tf.reshape(conv2_2,[FLAGS.batch_size,FLAGS.view_num,height/4,width/4,channel])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        radius=[32,16,8]
        xflow=tf.zeros([batch_size,height/4,width/4,1])
        cams=update_cams(cams,0.25)
        xflow,_=flow_pipline(ref_feature,view_features,cams,xflow,radius[0],conv_gru0,state0,64,0)
        up_flow=tf.image.resize_images(xflow,(height/2,width/2))
        # up_flow=tf.reshape(up_flow,[batch_size,height/2,width/2,2])
        conv2_2=tf.nn.relu(conv2_2)
        dconv3_0=deconv_bn(conv2_2,64,3,2,'SAME',reuse=tf.AUTO_REUSE,name='dconv3_0')
        conv3_1=conv_bn(tf.concat([dconv3_0,conv1_2],-1),32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_1')
        conv3_2=conv_bn(conv3_1,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_2')
        conv3_3=conv(conv3_1,32,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv3_3')
        features=tf.reshape(conv3_3,[FLAGS.batch_size,FLAGS.view_num,height/2,width/2,32])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        cams=update_cams(cams,2)
        flow,_=flow_pipline(ref_feature,view_features,cams,up_flow,radius[1],conv_gru1,state1,32,1)
        up_flow=tf.image.resize_images(flow,(height,width))
        # up_flow=tf.reshape(up_flow,[batch_size,height,width,2])
        conv3_3=tf.nn.relu(conv3_3)
        dconv4_0=deconv_bn(conv3_3,32,1,2,'SAME',reuse=tf.AUTO_REUSE,name='dconv4_0')
        conv4_1=conv_bn(tf.concat([dconv4_0,conv0_2],-1),16,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_1')
        conv4_2=conv_bn(conv4_1,16,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_2')
        conv4_3=conv(conv4_2,16,3,1,'SAME',reuse=tf.AUTO_REUSE,name='conv4_3')
        features=tf.reshape(conv4_3,[FLAGS.batch_size,FLAGS.view_num,height,width,16])
        ref_feature=tf.squeeze(tf.slice(features,[0,0,0,0,0],[-1,1,-1,-1,-1]),1)
        view_features=tf.slice(features,[0,1,0,0,0],[-1,-1,-1,-1,-1])
        cams=update_cams(cams,2)
        flow,depth=flow_pipline(ref_feature,view_features,cams,up_flow,radius[2],conv_gru2,state2,16,2)
        # up_flow=tf.image.resize_images(current_flow,(height,width))
        return depth
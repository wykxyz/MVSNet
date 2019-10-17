from __future__ import print_function
import os
import shutil
import cv2

import time
import glob
import random
import math
import re
import sys
import numpy as np
import tensorflow as tf
import scipy.io
import urllib
from tensorflow.python.lib.io import file_io

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()



#depth_path='/home/wzz/000015/'
depth_path='/xdata/wuyk/dtu_pytorch_test/yhw_dtu_RMVS_point_cloud/yhw-rmvs-depth/'
#image_path='/home/wzz/dtu/Rectified/'
#cam_path='/home/wzz/dtu/Cameras/train/'
#mvsnet_path='/home/wzz/MVSNet/mvsnet/depthfusion.py'
#fusibile_path='/home/haibao637/data/fusibile/build/fusibile'
#fuse_folder='/home/wzz/fuse_folder/'
#point_path='/home/wzz/points/'
path_all=os.listdir(depth_path)
#if not os.path.exists(point_path):
#    os.mkdir(point_path)
print(path_all)
for path_t in path_all:
    for i in range(49):
        file=open(depth_path+path_t+'/depths_mvsnet'+'/%08d_init.pfm'%(i+1))
        data=load_pfm(file)
        data=cv2.pyrUp(data)
        data=cv2.pyrUp(data)
        file.close()
        write_pfm(depth_path+path_t+'/depths_mvsnet'+'/%08d_init.pfm'%(i+1),data)

        file=open(depth_path+path_t+'/depths_mvsnet'+'/%08d_prob.pfm'%(i+1))
        data=load_pfm(file)
        data=cv2.pyrUp(data)
        data=cv2.pyrUp(data)
        file.close()
        write_pfm(depth_path+path_t+'/depths_mvsnet'+'/%08d_prob.pfm'%(i+1),data)
        
        img=cv2.imread(depth_path+path_t+'/depths_mvsnet'+'/%08d.jpg'%(i+1))
        img=cv2.pyrUp(img)
        img=cv2.pyrUp(img)
        cv2.imwrite(depth_path+path_t+'/depths_mvsnet'+'/%08d.jpg'%(i+1),img)
        cv2.imwrite(depth_path+path_t+'/images'+'/%08d.jpg'%(i+1),img)
        
        
          
"""    
for path_t in path_all:
    if not os.path.exists(fuse_folder+path_t):
        os.mkdir(fuse_folder+path_t)
    if not os.path.exists(fuse_folder + path_t+'/images/'):
        os.mkdir(fuse_folder + path_t+'/images/')
    if not os.path.exists(fuse_folder + path_t+'/cams/'):
        os.mkdir(fuse_folder + path_t+'/cams/')
    if not os.path.exists(fuse_folder + path_t+'/depths_mvsnet/'):
        os.mkdir(fuse_folder + path_t+'/depths_mvsnet/')
    if not os.path.exists(fuse_folder + path_t+'/points_mvsnet/'):
        os.mkdir(fuse_folder + path_t+'/points_mvsnet/')
    for i in range(49):
         shutil.copy(depth_path+path_t+'/init_depth_map_%04d.pfm'%i,fuse_folder + path_t+'/depths_mvsnet/')
         os.rename(fuse_folder + path_t+'/depths_mvsnet/'+'/init_depth_map_%04d.pfm'%i,fuse_folder + path_t+'/depths_mvsnet/'+'%08d_init.pfm'%i)
         file=open(fuse_folder + path_t+'/depths_mvsnet/'+'%08d_init.pfm'%i)
         data=load_pfm(file)
         data=cv2.pyrUp(data)
         data=cv2.pyrUp(data)
         file.close()
         write_pfm(fuse_folder + path_t+'/depths_mvsnet/'+'%08d_init.pfm'%i,data)
         shutil.copy(depth_path+path_t+'/init_depth_map_%04d.pfm' % i, fuse_folder + path_t + '/depths_mvsnet/')
         os.rename(fuse_folder + path_t + '/depths_mvsnet/' + '/init_depth_map_%04d.pfm'%i, fuse_folder + path_t+'/depths_mvsnet/'+'%08d_prob.pfm'%i)
         file = open(fuse_folder + path_t+'/depths_mvsnet/'+'%08d_prob.pfm'%i)
         data = load_pfm(file)
         data = cv2.pyrUp(data)
         data = cv2.pyrUp(data)
         file.close()
         write_pfm(fuse_folder + path_t+'/depths_mvsnet/'+'%08d_prob.pfm'%i, data)
         img=cv2.imread(image_path+path_t+'/rect_%03d_3_r5000.png'%(i+1))
         # img=cv2.pyrDown(img)
         # img=cv2.pyrDown(img)
         cv2.imwrite(fuse_folder+path_t+'/images/'+'%08d.jpg'%i,img)
         cv2.imwrite(fuse_folder+path_t+'/depths_mvsnet/'+'%08d.jpg'%i,img)
         shutil.copy(cam_path+'%08d_cam.txt'%i,fuse_folder+path_t+'/cams/')
         os.rename(fuse_folder+path_t+'/cams/'+'%08d_cam.txt'%i,fuse_folder+path_t+'/cams/'+'%08d.txt'%i)
         file=open(fuse_folder+path_t+'/cams/'+'%08d.txt'%i)
         shutil.copy(cam_path + '%08d_cam.txt' % i, fuse_folder + path_t + '/depths_mvsnet/')
         os.rename(fuse_folder + path_t + '/depths_mvsnet/' + '%08d_cam.txt'%i, fuse_folder + path_t + '/depths_mvsnet/' + '%08d.txt'%i)
    os.system('CUDA_ViSIBLE_DEVICES=1 python '+mvsnet_path+' --dense_folder '+fuse_folder+path_t+ '/ --fusibile_exe_path '+fusibile_path+' --prob_threshold 0.0')
    path_point=os.listdir(fuse_folder+path_t+'/points_mvsnet/')
    for path_s in path_point:
        if 'consistencyCheck' in path_s:
            shutil.copy(fuse_folder+path_t+'/points_mvsnet/'+path_s+'/final3d_model.ply',point_path)
            num=path_t.split('_')
            num2=num[0].split('n')
            num_str=num2[1]
            if (len(num_str)==1):
                num_str='00'+num_str
            else :
                if (len(num_str)==2):
                    num_str='0'+num_str
            os.rename(point_path+'final3d_model.ply',point_path+'yhw'+num_str+'_l3.ply')
            break
"""


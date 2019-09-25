import os
import time
import glob
import random
import math
import re
import sys
from struct import *
import cv2
import numpy as np
import pylab as plt
from tensorflow.python.lib.io import file_io
import matplotlib.image as mpimg

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

path_all=os.listdir('/home/yanjianfeng/data/wu-ETH3D-training/ETH3D_results/')
print path_all
for path in path_all:
    file1='/home/yanjianfeng/data/wu-ETH3D-training/ETH3D_results/'+path+'/gt_depth'
    print file1
    num=len(os.listdir(file1))
    num=num/3
    print num
    for i in range(num):
        file2=open(file1+'/%04d.npy'%i)
        init_depth_map_path = file1 + ('/%08d_init.pfm'%i)
        print file2
        im=np.load((file1+'/%04d.npy'%i))
        print im.shape
        write_pfm(init_depth_map_path, im)
    ##imgNum=len(file1)
    ##print imgNum

##init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
##write_pfm(init_depth_map_path, out_init_depth_image)


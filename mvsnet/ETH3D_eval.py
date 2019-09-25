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
        ##print width,height
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

path_all=os.listdir('/home/yanjianfeng/data/wu-ETH3D-training/ETH3D_results/')
print(path_all)
Tsum=0
for path in path_all:
    sum=0
    file1='/home/yanjianfeng/data/wu-ETH3D-training/ETH3D_results/'+path+'/gt_depth'
    ##print file1
    num=len(os.listdir(file1))
    num=num/4
    for i in range(num):
        file2=open('/home/yanjianfeng/data/wu-ETH3D-training/ETH3D_results/'+path+'/gt_depth'+'/%08d_init.pfm'%i)
        data2=load_pfm(file2)
        ##im2=np.load(file2)
        ##print im2.shape
        file3=open('/home/yanjianfeng/data/wu-ETH3D-training/ETH3D/'+path+'/depths_mvsnet'+'/%08d_init.pfm'%i)
        data3=load_pfm(file3)
        ##im3=np.load(file3)
        ##print im3.shape
        mask=(data2>0.5)&(data2<10)
        print(str(path)+' '+str(i)+' '+str(abs(data2[mask]-data3[mask]).mean()))
        sum+=abs(data2[mask]-data3[mask]).mean()
    sum=sum/num
    print(str(path)+' total '+str(sum))
    Tsum+=sum
Tsum=Tsum/13.0
print('all ' + str(Tsum))

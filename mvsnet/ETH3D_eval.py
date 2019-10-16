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
#import pylab as plt
from tensorflow.python.lib.io import file_io
import torch

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

def compute_errors_test(gt, pred):
    #gt=gt.numpy()
    #pred=pred.numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3

path_all=os.listdir('/data4/wyk_data/wu-ETH3D-training/ETH3D_results/')
print(path_all)
Tsum=0
Max=0
abs_rel_tot=0; abs_diff_tot=0; sq_rel_tot=0;rmse_tot=0; rmse_log_tot=0; a1_tot=0; a2_tot=0; a3_tot=0
for path in path_all:
    sum=0
    abs_rel_temp=0; abs_diff_temp=0; sq_rel_temp=0; rmse_temp=0; rmse_log_temp=0; a1_temp=0; a2_temp=0; a3_temp=0
    file1='/data4/wyk_data/wu-ETH3D-training/ETH3D_results/'+path+'/gt_depth'
    ##print file1
    num=len(os.listdir(file1))
    num=num/4
    for i in range(num):
        file2=open('/data4/wyk_data/wu-ETH3D-training/ETH3D_results/'+path+'/gt_depth'+'/%08d_init.pfm'%i)
        data2=load_pfm(file2)
        ##print "data2"
        ##print data2
        ##im2=np.load(file2)
        ##print im2.shape
        file3=open('/data4/wyk_data/wu-ETH3D-training/ETH3D/'+path+'/depths_mvsnet'+'/%08d_init.pfm'%i)
        data3=load_pfm(file3)
        ##print "data3"
        ##print data3
        ##im3=np.load(file3)
        ##print im3.shapie
        """
        width,height=data2.shape
        for x in range(width):
            for y in range(height):
                np.where(np.isnan(data2,0.0,data2[x][y])
        """
        data2=np.where(np.isnan(data2),0.0,data2)
        ###print "*****"
        ###print data2.mean()
        mask=(data2>0.0)&(data2<3.12)
        ##print("gt-max,gt-min")
        ##print(np.max(data2), np.min(data2))
        ##print("+++++")
        ##Max=max(np.max(data2),Max)
        """
        if np.isnan(data2[mask].mean()):
            print "//////"
            print data2
            exit(0)
        """
        ##print mask
        ##print(str(path)+' '+str(i)+' '+str(abs(data2[mask]-data3[mask]).mean()))
        sum+=abs(data2[mask]-data3[mask]).mean()
        
        tensor2=torch.Tensor(data2)
        tensor3=torch.Tensor(data3)
        gt=tensor2.numpy()
        gt=gt[mask]   
        pred=tensor3.numpy()
        pred=pred[mask]
        abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3=compute_errors_test(gt, pred)
        abs_rel_temp+=abs_rel;abs_diff_temp+=abs_diff;sq_rel_temp+=sq_rel;rmse_temp+=rmse;rmse_log_temp+=rmse_log;a1_temp+=a1;a2_temp+=a2;a3_temp+=a3
    sum=sum/num
    abs_rel_temp/=num; abs_diff_temp/=num; sq_rel_temp/=num; rmse_temp/=num; rmse_log_temp/=num; a1_temp/=num; a2_temp/=num; a3_temp/=num
    print(str(path)+' total '+str(sum))
    Tsum+=sum
    abs_rel_tot+=abs_rel_temp; abs_diff_tot+=abs_diff_temp; sq_rel_tot+=sq_rel_temp;rmse_tot+=rmse_temp; rmse_log_tot+=rmse_temp; a1_tot+=a1_temp; a2_tot+=a2_temp; a3_tot+=a3_temp
Tsum=Tsum/13.0
print('all ' + str(Tsum))
print('abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3',abs_rel_tot/13.0, abs_diff_tot/13.0, sq_rel_tot/13.0,rmse_tot/13.0, rmse_log_tot/13.0, a1_tot/13.0, a2_tot/13.0, a3_tot/13.0)
##print(Max)

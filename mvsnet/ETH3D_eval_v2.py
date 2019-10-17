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
    ##gt=gt.numpy()
    ##pred=pred.numpy()
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


output_depth='/data4/wyk_data/wu-ETH3D-training/output_depth_result/'
gt_depth='/data4/wyk_data/wu-ETH3D-training/gt_result/'
scale_result='/data4/wyk_data/wu-ETH3D-training/scale_result/'
maxdepth=10.0; mindepth=0.5
abs_rel_tot=0; abs_diff_tot=0; sq_rel_tot=0;rmse_tot=0; rmse_log_tot=0; a1_tot=0; a2_tot=0; a3_tot=0
total_num=len(os.listdir(gt_depth))

for i in range(total_num):
    tgt_depth=np.load(gt_depth+'%08d.npy'%i)
    pred=np.load(output_depth+'%08d.npy'%i)
    scale=open(scale_result+'%08d.txt'%i).read()
    scale_float=float(scale)
    mask=(tgt_depth<=maxdepth)&(tgt_depth>mindepth)&(tgt_depth==tgt_depth)
    abs_rel_temp,abs_diff_temp,sq_rel_temp,rmse_temp,rmse_log_temp,a1_temp,a2_temp,a3_temp=compute_errors_test(tgt_depth[mask]/scale_float, pred[mask]/scale_float)
    abs_rel_tot+=abs_rel_temp; abs_diff_tot+=abs_diff_temp; sq_rel_tot+=sq_rel_temp;rmse_tot+=rmse_temp; rmse_log_tot+=rmse_temp; a1_tot+=a1_temp; a2_tot+=a2_temp; a3_tot+=a3_temp

print('abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3')
print(abs_rel_tot/total_num, abs_diff_tot/total_num, sq_rel_tot/total_num,rmse_tot/total_num, rmse_log_tot/total_num, a1_tot/total_num, a2_tot/total_num, a3_tot/total_num)


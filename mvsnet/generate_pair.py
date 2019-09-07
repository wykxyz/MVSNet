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


def main(_):  # pylint: disable=unused-argument
    """ program entrance """
    # generate input path list
    mvs_list = gen_pipeline_mvs_list(FLAGS.dense_folder)
    for item in mvs_list:
        
    # mvsnet inference
    

if __name__ == '__main__':
    tf.app.run()

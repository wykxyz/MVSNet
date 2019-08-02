#-*- coding:utf-8 -*-
import re

# from preprocess import load_cam, write_pfm, load_pfm
import sys

import cv2
import numpy as np


def best_depth_range(ref_cam, view_cam, height, width,min_d,max_d):
    ref_k = ref_cam[1, :3, :3]
    ref_p = ref_cam[0, :, :]
    view_k = view_cam[1, :3, :3]
    view_p = view_cam[0, :, :]
    left_part = np.matmul(view_k, view_p[:3, :3])
    ref_c = -np.matmul(ref_p[:3, :3].transpose(), ref_p[:3, 3])
    view_c = -np.matmul(view_p[:3, :3].transpose(), view_p[:3, 3])
    mid_part = np.matmul((view_c - ref_c).reshape([3, 1]), ref_p[2, :3].reshape([1, 3]))
    # print mid_part.shape
    #     mid=(ref_p[:3,3]-view_p[:3,3]).dot(ref_p[3,:3])
    right_part = np.linalg.inv(np.matmul(ref_k, ref_p[:3, :3]))
    borders = [
        [0, 0, 1],
        [height, 0, 1],
        [height, width, 1],
        [0, width, 1]

    ]
    areas = height * width
    borders = np.array(borders).reshape([-1, 3, 1])  # 4,3

    def homo(d):
        h = np.matmul(np.matmul(left_part, np.eye(3, dtype=np.float32) - mid_part / d), right_part)
        new_border = np.squeeze(np.matmul(h[np.newaxis], borders), -1)  # 4,3
        new_border = new_border[..., :2] / new_border[..., -1:]  # 4,2
        #         print new_border
        x = new_border[:, 0]
        x = np.clip(x, 0, width)
        y = new_border[:, 1]
        y = np.clip(y, 0, height)
        polygon = np.stack([y, x], -1).astype(np.int32).reshape([1, -1, 2])
        #         print polygon.shape
        #         polygon = np.array([[[2, 2], [6, 2], [6, 6], [2, 6]]], dtype=np.int32)
        im = np.zeros([height, width], dtype="uint8")  # 获取图像的维度: (h,w)=iamge.shape[:2]
        polygon_mask = cv2.fillPoly(im, polygon, 255)
        # vis.image(polygon_mask)
        #         print im.shape
        #         polygon_mask = cv2.fillPoly(im, polygon, 1)
        #         vis.image(polygon_mask)
        area = np.sum(np.greater(polygon_mask, 0))
        #         maxs=np.max(new_border,0)#max_x,max_y
        #         mins=np.min(new_border,0)#min_x,min_y
        #         new_area=(y.max()-y.min())*(x.max()-x.min())

        return area

    def binary_search(dmin, dmax, op):
        if dmax - dmin <= 1e-2:
            print dmin, dmax
            return dmin
        dmean = (dmax - dmin) / 2
        #         print dmin,dmean,dmax
        new_area = homo(dmean)
        #         print new_area
        #         print areas*0.25-new_area
        if op(new_area, areas):
            return binary_search(dmin, dmean - 1e-2, op)
        else:
            return binary_search(dmean + 1e-2, dmax, op)

            #     dmax = binary_search(min_depth, max_depth, lambda a, b: a < b/4.0)
            #     dmin=binary_search( min_depth, max_depth, lambda a, b: a < b *0.9)
    old_area=None
    dmin=None
    # dmax=None
    for d in np.linspace(min_d, max_d, 50):
        # d=0.001+idx*0.1
        area = homo(d)
        # print d * 0.005, homo(d * 0.005)
        if area > areas / 5 and dmin  is None:
            dmin=d
            break

    if dmin is None:
        dmin=max_d
    # max_area=homo(10)
    # for d in  np.linspace(9.0,dmin+0.2,100):
    #     area=homo(d)
    #     if area!=max_area:
    #         dmax=d
    #         break
    # if dmax is None:
    #     dmax=1.0
    return dmin
    # return 0.2

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 256
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

from scipy.linalg import solve
def depth(x,x0,k1,k2,P1,P2):
    inv_k1=np.linalg.inv(k1)
    inv_k2=np.linalg.inv(k2)
    R1=P1[:3,:3]
    R2=P2[:3,:3]
    T1=P1[:3,3]
    T2=P2[:3,3]
    c1=-np.matmul(R1.transpose(),T1)
    c1=np.concatenate((c1,[1]),0)
    c2=-np.matmul(R2.transpose(),T2)
    c2=np.concatenate((c2,[1]),0)
    c=-np.matmul(P1,c1-c2)
    def func(x,x0):
        x0=np.matmul(inv_k1,x0)
        P=np.matmul(P1[:3,:],np.linalg.pinv(P2[:3,:]))
        x=np.matmul(inv_k2,x)
        x=-np.matmul(P,x)
    #     print x,x0
        a = np.array([[x0[0], x[0]], [x0[1], x[1]]])
        b = np.array([c[0],c[1]])
        out = solve(a, b)
        # print out
        return out[0]
    return func(x,x0)

def write_cam(file, cam):
    # f = open(file, "w")
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = str(file.readline()).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    # data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
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
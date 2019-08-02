# -*- coding:utf-8 -*-
import imageio
import numpy as np

"""
[x*y,3] * projH -> (x*y,new 3] * projH1 -> [x*y,re 3]
distance between [x*y,3] and [x*y,re 3]



"""

import os
import sys
# gpus = ["0"]
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import visdom
import torch
import torchvision
import torch.utils.data.dataset as dataset
vis = visdom.Visdom(env="MVSNet")
sys.path.append("../")
import re

# from preprocess import load_cam, write_pfm, load_pfm

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

def interpolate(image, x, y, dst_shape=None):
    image = image.permute(0, 2, 3, 1)
    batch_size, height, width,channels = image.shape
    dst_shape=dst_shape if dst_shape is not None else image.shape
    # image_shape=

    x = x - 0.5
    y = y - 0.5
    x0 = torch.floor(x).type(torch.long)
    x1 = x0 + 1
    y0 = torch.floor(y).type(torch.long)
    y1 = y0 + 1
    max_y = height - 1
    max_x = width - 1
    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)  # b,h,w
    y1 = torch.clamp(y1, 0, max_y)  # b,h,w
    mul = dst_shape[0] / batch_size
    b = torch.cat([torch.arange(batch_size).type(torch.long).reshape(-1, 1)] * mul * height * width, 1).reshape(-1,
                                                                                                                height*width).cuda()
    indices_a = torch.stack([b, y0, x0], dim=-1)  # b,h,w,2
    indices_b = torch.stack([b, y1, x0], dim=-1)  # b,h,w,2
    indices_c = torch.stack([b, y0, x1], dim=-1)  # b,h,w,2
    indices_d = torch.stack([b, y1, x1], dim=-1)  # b,h,w,2

    pixel_values_a = image[indices_a[..., 0], indices_a[..., 1], indices_a[..., 2]].cuda()
    pixel_values_b = image[indices_b[..., 0], indices_b[..., 1], indices_b[..., 2]].cuda()
    pixel_values_c = image[indices_c[..., 0], indices_c[..., 1], indices_c[..., 2]].cuda()
    pixel_values_d = image[indices_d[..., 0], indices_d[..., 1], indices_d[..., 2]].cuda()
    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)
    area_a = torch.unsqueeze(((y1 - y) * (x1 - x)), -1)  # b,h,w,1
    area_b = torch.unsqueeze(((y1 - y) * (x - x0)), -1)
    area_c = torch.unsqueeze(((y - y0) * (x1 - x)), -1)
    area_d = torch.unsqueeze(((y - y0) * (x - x0)), -1)
    output = area_a * pixel_values_a + area_b * pixel_values_b + area_c * pixel_values_c + area_d * pixel_values_d
    output=output.reshape([-1,height,width,channels]).permute(0,3,1,2).clamp(0,1)#b,c,h,w
    return output



def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = torch.linspace(0.5, width - 0.5, width)
    y_linspace = torch.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
    x_coordinates = torch.reshape(x_coordinates, [-1])
    y_coordinates = torch.reshape(y_coordinates, [-1])
    ones = torch.ones_like(x_coordinates)
    indices_grid = torch.cat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid



class PostNet(torch.nn.Module):
    def __init__(self,size=3):
        super(PostNet, self).__init__()
        self.size=size
        weight = torch.zeros([size * size, 1, size, size], dtype=torch.float32)
        for i in range(size * size):
            weight[i, 0, i // size, i % size] = 1
        self.CNN = torch.nn.Conv2d(1, self.size * self.size, self.size, 1,5,5)
        self.CNN.weight.data.copy_(weight)
        self.CNN=self.CNN.cuda()
        pass

    def forward(self, ref_color_maps, view_color_maps,ref_depth_maps, ref_cams, view_cams):
        """
sys.path.append("../")
        :param ref_depth_map:b,1,h,w
        :param view_depth_maps:n,b,1,h,w
        :param ref_cams:b,2,4,4
        :param view_cams: n,b,2,4,4
        :return:
        """

        view_num,batch_size,channel,height,width=view_color_maps.shape
        batch_size, _,height, width = ref_depth_maps.shape
        coords = torch.ones([height * width, 3])
        y, x = torch.meshgrid([torch.arange(height), torch.arange(width)])
        coords[..., 0] = x.reshape(-1)
        coords[..., 1] = y.reshape(-1)
        coords.unsqueeze_(0)  # 1,h*w,3
        ref_depth_map_vector = torch.reshape(ref_depth_maps, [batch_size, -1, 1])  # b,h*w
        ref_r = ref_cams[:, 0, :3, :3]  # b,3,3
        ref_k = ref_cams[:, 1, :3, :3]  # b,3,3
        ref_inv_k = []
        for i in range(batch_size):
            ref_inv_k.append(torch._np.linalg.inv(ref_k[i]))
        ref_inv_k = torch.Tensor(np.stack(ref_inv_k, 0)).unsqueeze(1)  # b,[s],3,3
        ref_inv_r = ref_r.permute(0, 2, 1).unsqueeze(1)  # b,[s],3,3
        ref_c = -1 * torch.matmul(ref_inv_r, ref_cams[:, 0, :3, 3].reshape(-1, 1, 3, 1))  # b,[s],3,1

        global_coords = torch.matmul(torch.matmul(ref_inv_r, ref_inv_k).cuda(),
                                     coords.unsqueeze(-1).cuda() * ref_depth_map_vector.unsqueeze(
                                         -1).cuda()) + ref_c.cuda()  # b,h*w,3,1
        global_coords.unsqueeze_(0)  # [n], b, h*w,3,1

        view_inv_r = view_cams[:, :, 0, :3, :3].permute(0, 1, 3, 2)  # n,b,3,3
        view_c = -1 * torch.matmul(view_inv_r, view_cams[:, :, 0, :3, 3].reshape(-1, batch_size, 3,
                                                                   1))  # 1,b,3,3  n,b,3,1 -> n,b,3,1
        view_c.unsqueeze_(2)  # n,b,[s],3,3
        view_r = view_cams[:, :, 0, :3, :3].unsqueeze(2)  # n,b,[s],3,3
        view_local_coords = torch.matmul(view_r.cuda(), (global_coords - view_c.cuda())).squeeze(-1)  # n,b,[s],3,1
        view_k = view_cams[:, :, 1, :3, :3].unsqueeze(2)#n,b,[s],3,3

        view_local_coords=torch.matmul(view_k.cuda(),view_local_coords.unsqueeze(-1)).squeeze(-1)
        del global_coords

        z = view_local_coords[..., 2] * 1.0

        view_local_coords = view_local_coords / view_local_coords[..., 2:] # n,b,h*w,3
        warp_x=view_local_coords[:,:,:,0]
        warp_y=view_local_coords[...,1]
        warp_views=[]
        view_features=self.CNN(view_color_maps.reshape([-1,channel,height,width]).cuda()).reshape([view_num,batch_size,-1,height,width])
        ref_feature=self.CNN(ref_color_maps.cuda()).unsqueeze(0)#1,b,c,h,w
        for i in range(view_num):
            warp_feature=interpolate(view_features[i],warp_x[i],warp_y[i])
            warp_views.append(warp_feature)
        warp_views=torch.stack(warp_views,0)
        ev=warp_views-torch.mean(warp_views,dim=2,keepdim=True)
        er=ref_feature-torch.mean(ref_feature,dim=2,keepdim=True)
        ncc_score=torch.mean(torch.mean(ev*er,dim=2,keepdim=True)/(torch.sqrt(torch.mean(ev**2,dim=2,keepdim=True)*torch.mean(er**2,dim=2,keepdim=True))+1e-7),dim=0)




        # ref_color_maps=ref_color_maps.cuda()
        #
        # expand_ref_color_maps=self.CNN(ref_color_maps)
        expand_ref_depth_maps=self.CNN(ref_depth_maps)
        # ds=torch.sum(2*torch.exp(-(expand_ref_color_maps-ref_color_maps)**2/10)*(ref_depth_maps-expand_ref_depth_maps),dim=1,keepdim=True)
        # loss0=torch.sum(torch.exp(-(expand_ref_color_maps-ref_color_maps)**2/10)*(expand_ref_depth_maps-ref_depth_maps)**2,dim=1,keepdim=True)
        # vis.images(loss0)
        # ref_depth_maps[...]=ref_depth_maps-1e-3*ds
        # loss=torch.sum(torch.exp(-(ref_feature-ref_color_maps)**2/10)*(expand_ref_depth_maps-ref_depth_maps)**2,dim=1,keepdim=True)
        # print torch.mean(loss0).item(),torch.mean(loss).item()
        return ncc_score



class PostDataset(dataset.Dataset):
    def __init__(self, dense_folder):
        image_folder = os.path.join(dense_folder, 'depths_mvsnet')
        self.cam_folder = image_folder
        self.depth_folder = image_folder
        cluster_list_path = os.path.join(dense_folder, 'pair.txt')
        cluster_list = open(cluster_list_path).read().split()

        # for each dataset
        self.mvs_list = []
        pos = 1
        for i in range(int(cluster_list[0])):
            idx = []
            # ref image
            ref_index = int(cluster_list[pos])
            idx.append(ref_index)
            pos += 1
            # ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
            # ref_cam_path = os.path.join(cam_folder, ('%08d.txt' % ref_index))
            # paths.append(ref_image_path)
            # paths.append(ref_cam_path)
            # view images
            all_view_num = int(cluster_list[pos])
            pos += 1
            check_view_num = min(3, all_view_num)
            for view in range(check_view_num):
                view_index = int(cluster_list[pos + 2 * view])
                idx.append(view_index)
                # paths.append(view_cam_path)
            pos += 2 * all_view_num
            # depth path
            self.mvs_list.append(idx)

        # self.mvs_list=self.mvs_list[1:]

    def __len__(self):
        return len(self.mvs_list)

    def __getitem__(self, item):
        data = self.mvs_list[item]  # save the idx
        cams = []
        depth_maps = []
        # depth_path = os.path.join(self.depth_folder, ('%08d_filter.pfm' % data[0]))
        # depth = load_pfm(open(depth_path))
        # for i in range(1, depth.shape[0]):
        #     for j in range(1, depth.shape[1]):
        #         if depth[i, j] == 0:
        #             depth[i, j] = np.max(depth[i - 1:i + 1, j - 1:j + 1])
        #
        # write_pfm(os.path.join(self.depth_folder, ("%08d_filter_1.pfm" % data[0])), depth)
        # print(data[0],'finished')
        # return depth
        color_maps = []
        for idx in data:
            cam_path = os.path.join(self.cam_folder, ('%08d.txt' % idx))
            depth_path = os.path.join(self.depth_folder, ('%08d_filter.pfm' % idx))
            color_path = os.path.join(self.depth_folder, ('%08d.jpg' % idx))
            cam = load_cam(open(cam_path, 'r'))
            cams.append(cam)
            depth_map = load_pfm(open(depth_path))
            depth_maps.append(depth_map)
            color_map = cv2.imread(color_path)
            color_map=cv2.cvtColor(color_map,cv2.COLOR_RGB2GRAY)
            color_map = color_map[...,np.newaxis] / 255.0
            color_map = color_map.transpose(2, 0, 1)
            color_maps.append(color_map)
        cams = np.stack(cams, axis=0).astype(np.float32)
        depth_maps = np.stack(depth_maps, axis=0).astype(np.float32)
        color_maps = np.stack(color_maps, axis=0).astype(np.float32)
        return cams, depth_maps, color_maps, data[0]


import cv2
from torch import optim
import torch.nn as nn
import argparse
def vis_show(group,*tensors):
    for index,tensor in enumerate(tensors):
        vis.images(tensor,win=group+" %d"%(index))
if __name__ == "__main__":
    # family horse fransic lighthouse m60 panther playground train
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, default='/home/haibao637/data/mvsnet_input//horse/')
    args = parser.parse_args()
    dense_folder = args.dense_folder
    postset = PostDataset(dense_folder)

    out_dir = os.path.join(dense_folder, 'depths_mvsnet')
    dataloader = torch.utils.data.DataLoader(postset, batch_size=1, num_workers=4)
    criternion=torch.nn.L1Loss()
    postnet = PostNet()
    for cams, depth_maps, color_maps, idx in dataloader:
        ref_cams = cams[:, 0, ...]
        view_cams = cams[:, 1:, ...].permute(1, 0, 2, 3, 4)

        # ref_cams[:,1, 0, 0] *= 2
        # ref_cams[:,1, 1, 1] *= 2
        ref_depth_maps = depth_maps[:, 0, ...].unsqueeze(1)#b,1,h,w
        ref_depth_maps.requires_grad=True
        optimizer = optim.Adam([ref_depth_maps], lr=1e-3)

        # print(ref_depth_maps.shape)
        print("refine ",idx)
        ref_color_maps = color_maps[:, 0, ...]#b,3,h,w
        view_color_maps = color_maps[:, 1:, ...].permute(1, 0, 2, 3, 4)#n,b,3,h,w
        ref_depth_maps=ref_depth_maps.cuda()
        postnet.eval()
        # vis_show("init",ref_depth_maps.detach().cpu().numpy())
        initial_score=postnet(ref_color_maps, view_color_maps, ref_depth_maps, ref_cams,
                                                view_cams)
        interval=ref_cams[:,1,3,2].reshape(ref_cams.shape[0],1,1,1)#b,1
        mask=initial_score<0.7
        ref_depth_maps[mask]=0
        initial_score[mask]=0
        vis_show("refine",
                 ref_depth_maps.detach().cpu().numpy(),
                 initial_score.detach().cpu().numpy()
                 )
        # for i in range(6):
        #     #mean-delta + (delta*T(2) * RAND())/RAND_MAX
        #     random_depth=ref_depth_maps-(interval+interval*2*torch.rand(1,1,ref_depth_maps.shape[2],ref_depth_maps.shape[3])).cuda()*ratios[i]
        #     score=postnet(ref_color_maps,view_color_maps,random_depth,ref_cams,view_cams)
        #     mask=score<initial_score
        #     ref_depth_maps[mask]=random_depth[mask]
        #     initial_score[mask]=score[mask]
        #     vis_show("refine",
        #              ref_depth_maps.detach().cpu().numpy(),
        #              initial_score.detach().cpu().numpy()
        #              )
            # ref_depth_maps=filter_depth_maps.detach()

            # loss.backward()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # del loss
            # vis_show("refine",
            #          filter_depth_maps.detach().cpu().numpy()
            #          )
            # del filter_depth_maps
        #     ref_depth_maps = filtered_depth_maps.detach()
        filtered_depth_maps=ref_depth_maps.detach().cpu().numpy()
        for i in range(ref_cams.shape[0]):
            print(idx[i].item())
            filtered_depth_map = filtered_depth_maps[i].squeeze()
            write_pfm(os.path.join(out_dir, "%08d_filter_1.pfm" % idx[i].item()), filtered_depth_map)

            write_cam(os.path.join(out_dir,"%08d.txt"%idx[i].item()), ref_cams[i].cpu().numpy())








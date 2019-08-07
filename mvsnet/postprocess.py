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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import visdom
import torch
import torch.utils.data.dataset as dataset
vis = visdom.Visdom(env="MVSNet")
sys.path.append("../")
from utils import *

def interpolate(image, x, y, dst_shape=None):
    # shape=shape if shape is not None else image.shape
    # image_shape=
    batch_size, channels, height, width = image.shape
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
                                                                                                                height,
                                                                                                                width).cuda()
    indices_a = torch.stack([b, y0, x0], dim=-1)  # b,h,w,2
    indices_b = torch.stack([b, y1, x0], dim=-1)  # b,h,w,2
    indices_c = torch.stack([b, y0, x1], dim=-1)  # b,h,w,2
    indices_d = torch.stack([b, y1, x1], dim=-1)  # b,h,w,2
    image = image.permute(0, 2, 3, 1)
    pixel_values_a = image[indices_a[..., 0], indices_a[..., 1], indices_a[..., 2]].squeeze().cuda()
    pixel_values_b = image[indices_b[..., 0], indices_b[..., 1], indices_b[..., 2]].squeeze().cuda()
    pixel_values_c = image[indices_c[..., 0], indices_c[..., 1], indices_c[..., 2]].squeeze().cuda()
    pixel_values_d = image[indices_d[..., 0], indices_d[..., 1], indices_d[..., 2]].squeeze().cuda()
    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)
    area_a = torch.unsqueeze(((y1 - y) * (x1 - x)), -1)  # b,h,w,1
    area_b = torch.unsqueeze(((y1 - y) * (x - x0)), -1)
    area_c = torch.unsqueeze(((y - y0) * (x1 - x)), -1)
    area_d = torch.unsqueeze(((y - y0) * (x - x0)), -1)
    output = area_a * pixel_values_a + area_b * pixel_values_b + area_c * pixel_values_c + area_d * pixel_values_d
    # output=output.reshape([-1,height,width,channels]).transpose(0,3,1,2)#b,c,h,w
    return output.permute(0, 3, 1, 2)


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


def homography_warping(input_image, homography, image_shape=None):
    image_shape = input_image.shape if image_shape is None else image_shape
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]

    # turn homography to affine_mat of size (B, 2, 3) and div_mat of size (B, 1, 3)
    # affine_mat = tf.slice(homography, [0, 0, 0], [-1, 2, 3])
    affine_mat = homography[:, :2, :3]
    # div_mat = tf.slice(homography, [0, 2, 0], [-1, 1, 3])
    div_mat = homography[:, 2:3, :3]
    # generate pixel grids of size (B, 3, (W+1) x (H+1))
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = pixel_grids.unsqueeze(0)
    # pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = pixel_grids.repeat(batch_size, 1).view(batch_size, 3, -1)
    # pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))
    # return pixel_grids

    # affine + divide tranform, output (B, 2, (W+1) x (H+1))
    grids_affine = torch.matmul(affine_mat, pixel_grids)
    grids_div = torch.matmul(div_mat, pixel_grids)
    grids_zero_add = torch.equal(grids_div, 0.0).type('float32') * 1e-7  # handle div 0
    grids_div = grids_div + grids_zero_add
    grids_div = grids_div.repeat(1, 1, 1)
    grids_inv_warped = torch.div(grids_affine, grids_div)
    x_warped, y_warped = torch.unbind(grids_inv_warped, dim=1)
    x_warped_flatten = torch.reshape(x_warped, [-1])
    y_warped_flatten = torch.reshape(y_warped, [-1])

    # interpolation
    warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten, image_shape)
    warped_image = torch.reshape(warped_image, shape=image_shape, name='warped_feature')

    # return input_image
    return warped_image


class PostNet(torch.nn.Module):
    def __init__(self,size=3):
        super(PostNet, self).__init__()
        self.size = size
        weight = torch.zeros([size * size, 1, size, size], dtype=torch.float32)
        for i in range(size * size):
            weight[i, 0, i // size, i % size] = 1
        self.CNN = torch.nn.Conv2d(1, self.size * self.size, self.size, 1, 1)
        self.CNN.weight.data.copy_(weight)
        self.CNN = self.CNN.cuda()

    def forward(self, ref_color_maps, view_color_maps, ref_depth_map, view_depth_maps, ref_cams, view_cams):
        """
sys.path.append("../")
        :param ref_depth_map:b,h,w
        :param view_depth_maps:n,b,h,w
        :param ref_cams:b,2,4,4
        :param view_cams: n,b,2,4,4
        :return:
        """

        view_num = view_depth_maps.shape[0]

        batch_size,_, height, width = ref_depth_map.shape
        coords = torch.ones([height * width, 3])
        y, x = torch.meshgrid([torch.arange(height), torch.arange(width)])
        coords[..., 0] = x.reshape(-1)
        coords[..., 1] = y.reshape(-1)
        coords.unsqueeze_(0)  # 1,h*w,3
        ref_depth_map_vector = torch.reshape(ref_depth_map, [batch_size, -1, 1])  # b,h*w
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
        del global_coords

        z = view_local_coords[..., 2] * 1.0

        view_local_coords = view_local_coords / view_local_coords[..., 2].unsqueeze(-1)  # n,b,h*w,3
        view_k = view_cams[:, :, 1, :3, :3].unsqueeze(2)  # n,b,[s],3,3
        view_coords = torch.matmul(view_k.cuda(), view_local_coords.unsqueeze(-1)).squeeze(-1)
        mask = (view_coords[..., 1] < 0) | (view_coords[..., 1] >= height) \
               | (view_coords[..., 0] < 0) | (view_coords[..., 0] >= width) | (z <= 0)

        view_coords = view_coords[..., 1].type(torch.long) * width + view_coords[..., 0].type(torch.long)  # n,b,h*w
        view_coords[mask] = 0
        # view_depth_maps
        view_depth_maps_vector = view_depth_maps.reshape(view_num, batch_size, -1)
        view_ref_depth_maps_vector = torch.gather(input=view_depth_maps_vector.cuda(), dim=-1,
                                                  index=view_coords)  # n,b,-1
        view_ref_depth_maps_vector[mask] = 1e-7
        del view_coords
        # vis.images(view_ref_depth_maps_vector.reshape(-1,1,height,width).numpy())
        view_inv_r.unsqueeze_(2)  # n,b,[s],3,3
        global_coords = torch.matmul(view_inv_r.cuda(),
                                     (view_local_coords * (view_ref_depth_maps_vector.unsqueeze(-1))).unsqueeze(-1)) \
                        + view_c.cuda()
        del view_ref_depth_maps_vector, view_local_coords
        ref_k.unsqueeze_(0).unsqueeze_(2)  # [n],b,[s],3,3
        ref_r.unsqueeze_(0).unsqueeze_(2)  # [n],b,[s],3,3
        ref_c.unsqueeze_(0)  # [n],b,[s],3,1

        new_coords = torch.matmul(torch.matmul(ref_k, ref_r).cuda(),
                                  global_coords - ref_c.cuda()).squeeze(-1)  # n,b,-1,3
        #
        #
        # alpha=view_ref_depth_maps_vector/z # n,b,h*w,1   1,b,h*w,1
        # alpha.unsqueeze_(-1).unsqueeze_(-1) # 1,b,-1,1,1
        #
        # # alpha n,b,-1,1 coords b,-1,3,1 d0 b,-1,1
        # new_coords=alpha*coords.unsqueeze(0)*(ref_depth_map_vector.unsqueeze(0).unsqueeze(-1))\
        #            +torch.matmul(ref_r.reshape(1,-1,1,3,3),((1-alpha)*(view_c-ref_c)))
        # coords=torch.matmul(ref_k,coords).squeeze(-1)
        # view_k = ref_cams[:, 1, :3, :3]
        # new_coords=torch.matmul(view_k,new_coords).squeeze(-1)
        z1 = new_coords[..., 2].reshape(-1, batch_size,1, height, width)
        z1[z1 < 0] = 0.0
        # vis.images(z1.numpy())
        # z1=torch.mean(z1,dim=0)
        # distance1=torch.abs(z1.reshape(batch_size,height,width)-ref_depth_map)
        new_coords[..., :2] = new_coords[..., :2] / new_coords[..., 2].unsqueeze(-1)
        distance2 = (new_coords[..., :2] - coords[..., :2].cuda()) ** 2
        ref_depth_map = ref_depth_map.cuda()
        mask = ((torch.sqrt(torch.sum(distance2, dim=-1)) < 1.0).reshape(-1, batch_size,1, height, width)) & (
        (torch.abs(z1 - ref_depth_map) *2/torch.abs(ref_depth_map+z1)< 0.01)) & (z1 > 0) & (ref_depth_map > 0.0)
        # mask=((torch.abs(z1-ref_depth_map)<0.01))
        mask = torch.sum(mask.type(torch.float32), dim=0) >= 3.0
        
        ref_color_maps = ref_color_maps.cuda()
        ref_depth_map=ref_depth_map.cuda()
        # expand_ref_color_maps = self.CNN(ref_color_maps)
        # expand_ref_depth_maps = self.CNN(ref_depth_map)
        # # ds = torch.sum( torch.exp(-(expand_ref_color_maps - ref_color_maps) ** 2 / 10) *
        # #             (ref_depth_map - expand_ref_depth_maps)**2, dim=1, keepdim=True)#b,1,h,w
        # e1=expand_ref_color_maps-torch.mean(expand_ref_color_maps,dim=1,keepdim=True)
        # e2=expand_ref_depth_maps-torch.mean(expand_ref_depth_maps,dim=1,keepdim=True)
        # sigma1=torch.sqrt(torch.mean(e1**2,dim=1,keepdim=True))
        # sigma2 = torch.sqrt(torch.mean(e2 ** 2, dim=1, keepdim=True))
        #
        # e1=e1/sigma1
        # e2=e2/sigma2
        # ds=torch.mean(torch.exp(-(e1-ref_color_maps)**2)*(e2-ref_depth_map)**2,dim=1,keepdim=True)

        # ds=torch.mean(e1*e2,dim=1,keepdim=True)/torch.sqrt(torch.mean(e1**2,dim=1,keepdim=True)*torch.mean(e2**2,dim=1,keepdim=True)+1e-7)

        # mask=mask&(ds>0.8)

        mask=mask.type(torch.float32)#b,h,w,1
      



        # vis.images((ds+1)/2*255.0)





        ref_depth_map = mask * ref_depth_map
        return ref_depth_map, mask


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
            check_view_num = min(8, all_view_num)
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
            depth_path = os.path.join(self.depth_folder, ('%08d_init.pfm' % idx))
            color_path = os.path.join(self.depth_folder, ('%08d.jpg' % idx))
            cam = load_cam(open(cam_path, 'r'))
            cams.append(cam)
            depth_map = load_pfm(open(depth_path))
            # depth_map=cv2.bilateralFilter(depth_map,10, 10 * 2, 10 / 2)
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
import argparse
if __name__ == "__main__":
    # family horse fransic lighthouse m60 panther playground train
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, default='/home/haibao637/data/mvsnet_input//horse/')
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()
    dense_folder = args.dense_folder
    postset = PostDataset(dense_folder)

    out_dir = os.path.join(dense_folder, 'depths_mvsnet')
    dataloader = torch.utils.data.DataLoader(postset, batch_size=args.batch_size, num_workers=4)

    postnet = PostNet().cuda()
    for cams, depth_maps, color_maps, idx in dataloader:
        ref_cams = cams[:, 0, ...]
        view_cams = cams[:, 1:, ...].permute(1, 0, 2, 3, 4)

        # ref_cams[:,1, 0, 0] *= 2
        # ref_cams[:,1, 1, 1] *= 2
        ref_depth_maps = depth_maps[:, :1, ...]#b,1,h,w
        # print(ref_depth_maps.shape)
        view_depth_maps = depth_maps[:, 1:, ...].permute(1, 0, 2, 3)
        ref_color_maps = color_maps[:, 0, ...]
        view_color_maps = color_maps[:, 1:, ...].permute(1, 0, 2, 3, 4)
        filtered_depth_maps, mask = postnet(ref_color_maps, view_color_maps, ref_depth_maps, view_depth_maps, ref_cams,
                                            view_cams)
        filtered_depth_maps = filtered_depth_maps.cpu()

        # vis.images(torch.stack([filtered_depth_maps.cpu(),ref_depth_maps.cpu()],0).unsqueeze(1))
        # vis.images(filtered_depth_maps.unsqueeze(1))
        filtered_depth_maps = filtered_depth_maps.cpu().numpy()

        for i in range(ref_cams.shape[0]):
            print(idx[i].item())
            filtered_depth_map = np.squeeze(filtered_depth_maps[i])
            write_pfm(os.path.join(out_dir, "%08d_filter.pfm" % idx[i].item()), filtered_depth_map)

            # write_cam(os.path.join(out_dir,"%08d.txt"%idx[i].item()), ref_cams[i].cpu().numpy())








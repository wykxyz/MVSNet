# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class NCC(torch.nn.Module):
    def __init__(self, vis, size=5):
        super(NCC, self).__init__()
        self.size = size
        self.vis = vis
        self.maxpool = torch.nn.MaxPool2d(kernel_size=size, stride=1, padding=(size - 1) // 2, return_indices=True)
        weight = torch.zeros([size * size, 1, size, size], dtype=torch.float32)
        for i in range(size * size):
            weight[i, 0, i // size, i % size] = 1
        self.CNN = torch.nn.Conv2d(1, self.size * self.size, self.size, 1, 1)
        self.CNN.weight.data.copy_(weight)
        self.scaleRanges = (
            1,
            0.5,
            0.25
            , 0.125
            , 0.0625
            , 0.03125
            , 0.015625
            , 0.0078125
            , 0.00390625
            , 0.001953125
            , 0.0009765625
            , 0.00048828125
        )

    def forward(self, src, dst, Ks, Rs, Ts, depth, norm_map, dilation, threshold=0.5, depth_size=None):
        if depth_size is None:
            depth_size = self.size
        iter = dilation.type(torch.int)[0].item()
        depth_size = depth_size.type(torch.int)[0].item()
        threshold = threshold[0].item()
        return self.ncc(src, dst, Ks, Rs, Ts, depth, norm_map, depth_size, iter, threshold)

    def ncc(self, src, dst, Ks, Rs, Ts, depth, norm_map, depth_size0, iter, threshold=0.5):
        """

        :param src:
        :param dst:
        :param Ks: b,n,2,3,3
        :param Rs: b,n,3,3
        :param Ts: b,n,3,3
        :param depth:
        :param norm_map:
        :param depth_size0:
        :param iter:
        :param threshold:
        :return:
        """

        width = src.shape[3]
        height = src.shape[2]
        channels = src.shape[1]
        view_nums = dst.shape[1]
        batch_size = src.shape[0]
        patch_size = self.size * self.size
        self.device = self.CNN.weight.device
        src = src.to(self.device)
        dst = dst.to(self.device)
        Ks = Ks.to(self.device)
        Rs = Rs.to(self.device)
        Ts = Ts.to(self.device)
        depth = depth.to(self.device)
        norm_map = norm_map.to(self.device)

        depth_size = depth_size0 * depth_size0
        depth_weight = torch.zeros([depth_size, 1, depth_size0, depth_size0], dtype=torch.float32).to(self.device)
        depth_weight[0, 0, depth_size0 // 2, depth_size0 // 2] = 1
        for i in range(0, depth_size):
            depth_weight[i, 0, i // depth_size0, i % depth_size0] = 1
        depth_weight[0, 0, 0] = 0
        depth_weight[0, 0, depth_size0 // 2, depth_size0 // 2] = 1.0
        depth_weight[depth_size // 2, 0, depth_size0 // 2, depth_size0 // 2] = 0
        depth_weight[depth_size // 2, 0, 0, 0] = 1.0
        depth_size = depth_size0 * depth_size0

        coord_buffer = np.ones((height, width, 3), dtype=np.float32)
        coord_buffer[..., 1], coord_buffer[..., 0] = np.mgrid[0:height, 0:width]
        coord_buffer = torch.Tensor(coord_buffer).to(self.device).permute(2, 0, 1).unsqueeze(1).to(
            self.device)  # b,3,h,w
        coord_buffer = F.conv2d(coord_buffer, weight=self.CNN.weight, padding=(self.size - 1) // 2).permute(1, 2, 3,
                                                                                                            0)  # p,h,w,3

        # P: b,4,4 *n,9,h,w,b,4
        # F : b,n,3,3, coord: 9,h,w,b,n,3,1
        # depth :b,9,h,w -> 9,h,w,b,1 # PC: B,3,1 ->  9,h,w,b,3-> b,9,h,w,3,1
        depth = F.conv2d(depth, depth_weight, padding=iter * (depth_size0 - 1) // 2,
                         dilation=iter)  # b,size*size,w,h   ex: b,d,h,w

        # ==================
        # # reprojection
        # # d,p,h,w,b,n,3
        # # K*R*K_inv*x+Kt/d
        # #K:b,n,2,3,3  t:b,n,[d],[h],[w],3,1  depth b,[n],d,h,w,[1],[1]
        # dt=(torch.matmul(Ks[:,:,1,...],Ts).reshape(batch_size,view_nums,1,1,1,3,1)*(1.0/depth).reshape(batch_size,1,depth_size,height,width,1,1))
        # P=torch.matmul(torch.matmul(Ks[:,:,1,...],Rs),Ks[:,:,0,...])#b,n,[d],[p],[h],[w],3,3
        # #coord_buffer [b],[n],p,h,w,3,[1]
        # coord_buffer=torch.matmul(P.reshape(batch_size,view_nums,1,1,1,3,3),coord_buffer.reshape(1,1,patch_size,height,width,3,1))
        # #coord_buffer b,n,[d],p,h,w,3,1 dt b,n,d,[p],h,w,3,1
        # coord_buffer=coord_buffer.reshape(batch_size,view_nums,1,patch_size,height,width,3,1)\
        #              +dt.reshape(batch_size,view_nums,depth_size,1,height,width,3,1)
        # coord_buffer.squeeze_(-1)
        # del dt,P
        # ===================
        # Homographing

        # PC 3*3 depth:b,n,d,p,h,w->d,p,h,w,b,n,1 *3,3
        # DT b,n,3,1  n: b,h,w,3->b,n,h,w,3 -> b,n,h,w,3,3
        # H=PP[1] * (I-(DT*N)/d)*PP[0]

        norm_map = F.conv2d(norm_map.reshape(-1, 1, height, width), depth_weight, padding=(depth_size0 - 1) // 2) \
            .reshape(batch_size, -1, depth_size, height, width).permute(0, 2, 3, 4, 1)  # b,d,h,w,1,3
        # Ts b,n,[d],[h],[w],3,1  norm: b,[],d,h,w,1,3 -> b,n,d,h,w,3,3
        m = torch.matmul(Ts.reshape(batch_size, view_nums, 1, 1, 1, 3, 1),
                         norm_map.reshape(batch_size, 1, depth_size, height, width, 1, 3))  # b,n,d,h,w,3,3
        # norm_map:b,n,d,h,w,3,3 depth: b,[1],d,h,w,[1],[1]
        m = m / (depth.reshape(batch_size, 1, depth_size, height, width, 1, 1))  # b,n,d,h,w,3,3
        m = torch.eye(3, 3).reshape(1, 1, 1, 1, 1, 3, 3).to(self.device) - m
        # PP b,n,2,3,3
        # PP[:,:,1]# b,n,3,3 norm_map:b,n,d,h,w,3,3
        # Rs b,n,3,3 Ks b,n,2,3,3
        PP = torch.matmul(Ks[:, :, 1, ...], Rs)
        H = torch.matmul(torch.matmul(PP.reshape(batch_size, view_nums, 1, 1, 1, 3, 3), m),
                         Ks[:, :, 0, ...].reshape(batch_size, view_nums, 1, 1, 1, 3, 3)).unsqueeze(3)  # b,n,d,1,h,w,3,3
        # coord_buffer p,h,w,3
        coord_buffer = torch.matmul(H, coord_buffer.reshape(1, 1, 1, patch_size, height, width, 3, 1)).squeeze(
            -1)  # b,n,d,p,h,w,3
        del H, m

        # =================================
        mask = coord_buffer[..., 2] > 0.0
        mask &= (norm_map.reshape(batch_size, 1, depth_size, 1, height, width, 3)[..., 2] < 0)
        coord_buffer[..., 0][mask] = coord_buffer[..., 0][mask] / coord_buffer[..., 2][mask]
        coord_buffer[..., 1][mask] = coord_buffer[..., 1][mask] / coord_buffer[..., 2][mask]
        mask = (depth.reshape(batch_size, 1, depth_size, 1, height, width) > 0) \
               & mask & ((coord_buffer[..., 0] > 0) & (coord_buffer[..., 0] < width - 1)) & (
                       (coord_buffer[..., 1] > 0) & (coord_buffer[..., 1] < height - 1))  # b,n,d,p,h,w
        # print(torch.sum(mask),'valid coords')
        coord_buffer[..., 0][mask == False] = 0
        coord_buffer[..., 1][mask == False] = 0
        coord_buffer[..., 2][mask == False] = 0

        # coord_buffer = coord_buffer.type(torch.long).detach()  # b,n,d,p
        """
        const int lx((int)pt.x);
    const int ly((int)pt.y);
    const T x(pt.x-lx), x1(T(1)-x);
    const T y(pt.y-ly), y1(T(1)-y);
    return (BaseBase::operator()( ly, lx)*(1-x) + BaseBase::operator()( ly, lx+1)*x)*(1-y) +
           (BaseBase::operator()(ly+1,lx)*(1-x) + BaseBase::operator()(ly+1,lx+1)*x)*y;"""

        dst = torch.stack([dst.permute(0, 1, 3, 4, 2)] * (patch_size * depth_size), 2).reshape(
            batch_size, view_nums, depth_size, patch_size, height, width, channels
        )  # b,n,d,p,h,w,c
        shape = dst.shape
        dst = dst.reshape(batch_size, view_nums, depth_size, patch_size, height * width, channels)  # b,n,d,p,h*w,c
        coords = coord_buffer.reshape(batch_size, view_nums, depth_size, patch_size, height * width,
                                      -1).type(torch.long)  # b,n,d,p,h,w,3,1 # b,n,d,p,h*w,3
        del coord_buffer
        # x =coords-(coords.floor())
        # _x=1.0-x

        coords = coords[..., 1] * width + coords[..., 0]
        coords = torch.stack([coords] * channels, -1)
        dst = torch.gather(dim=4, index=coords.type(torch.long), input=dst)
        dst = dst.reshape(shape)  # b,n,d,p,h,w,c
        dst[mask == False, :] = 0.0
        # print(dst.min(),dst.max())
        # return dst.reshape(-1, channels, height, width)

        # ================expand src======================
        src = F.conv2d(src, weight=self.CNN.weight, padding=(self.size - 1) // 2)  # b,p,h,w
        src_ = src.permute(0, 2, 3, 1)  # b,h,w,p
        dst_ = dst.squeeze(-1).permute(1, 2, 0, 4, 5, 3)  # b,n,d,p,h,w -> n,d,b,h,w,p
        # above  src  src: b,b,h,w,p dst :n,d,b,h,w,p
        # ==================end expand src================
        # =============ncc score==========================
        src_ = src_ - torch.mean(src_, dim=-1, keepdim=True)
        dst_ = dst_ - torch.mean(dst_, dim=-1, keepdim=True)
        norm = torch.sqrt(torch.mean(src_ ** 2, dim=-1, keepdim=True) * torch.mean(dst_ ** 2, dim=-1, keepdim=True))

        # ============consistance============

        # b,n,d,p,h,w
        # depth b,n,h,w,p,d  b,n,h,w,d,p
        # p1d1,p1d2,p1d3 d1p1,d1p2,d1p3
        # p1d1d1p1

        # mask: b,n,d,p,h,w ->
        # dst:n,d,b,h,w,p
        # depth b,n,d,p,h,w
        # d0=depth.permute(2,0,3,4,-1,1)#b,n,h,w,p,d
        # ones=torch.ones(batch_size,view_nums,height,width,depth_size,patch_size,1).to(self.device)
        # ddst=torch.matmul(ones,d0.unsqueeze(-2))#b,n,h,w,p,d,d
        # ddst=(ddst+d0.unsqueeze(-1))/(ddst-d0.unsqueeze(-1)) # b,n,h,w,p,d,d
        # m0=mask.permute(0,1,4,5,3,2)
        # emask=torch.matmul(ones,m0.unsqueeze(-2))*(m0.unsqueeze(-1))
        # ddst[emask==False]=2e-3
        # m0=m0==False
        # emask = torch.matmul(ones, m0.unsqueeze(-2)) * (m0.unsqueeze(-1))
        # ddst[emask]=0.0
        # ddst=torch.mean(ddst,dim=-1) # b,n,,w,p,d
        #
        # del d0
        # dst=dst.permute(1,-2,0,2,3,-1)#n,d,b,h,w,p
        # ===================================
        dst_ = torch.mean(src_ * dst_ / norm, dim=-1)  # #n,d,b,h,w -> b,n,d,h,w # discard p
        norm.squeeze_(-1)
        m = norm <= 0
        # dst = dst / norm
        dst_[m] = threshold
        # m=m==False
        # print(dst[m].min().item(),dst[m].mean().item(),dst[m].max().item())
        del m, norm
        dst_ = dst_.permute(2, 0, 1, 3, 4)

        # return dst.reshape(-1, channels, height, width)

        mask = (mask)  # b,n,d,p,h,w
        mask = torch.all(mask, dim=3)  # each pixel is in bound  #,b,d,h,w
        score = torch.ones(mask.shape).type(torch.float32).to(self.device) * threshold  # b,n,d,h,w
        score[mask] = dst_[mask]
        # score *= (1 - 0.1 * torch.exp(-
        #                               torch.sum((depth[:, :, :, 0, ...] - torch.mean(depth[:, :, :, 0, ...], dim=2,
        #                                                                              keepdim=True)) ** 2 / (
        # 	                                        2 * 0.006 * 0.006), dim=2, keepdim=True)
        #                               ))  # b,n,d,h,w
        score = torch.mean(score, dim=1)  # b,d,h,w



        mask = torch.any(mask, dim=1)
        mask = mask & (depth > 0)
        score[mask == False] = threshold
        # ================end of nccscore==================
        prob, index = torch.max(score, dim=1, keepdim=True)
        maxs, _ = torch.max(mask, dim=1, keepdim=True)
        mins, _ = torch.min(mask, dim=1, keepdim=True)
        dst = dst[:, :, :, dst.shape[3] // 2, ..., 0]  # b,d,h,w

        dst_index = torch.stack([index] * view_nums, dim=1)
        dst = torch.gather(dst, index=dst_index, dim=2)
        mask = torch.gather(mask, index=index, dim=1)
        depth = torch.gather(depth, index=index, dim=1)  # b,1,h,w
        norm_index = torch.stack([index] * 3, dim=-1)
        norm_map = torch.gather(norm_map, index=norm_index, dim=1)

        # print(torch.any(torch.isinf(norm_map)|torch.isnan(norm_map)))
        # mask=mask&(torch.any(torch.isnan(norm_map))==False)

        # d1=F.conv2d(depth,self.CNN.weight,padding=(self.size-1)//2)# b,d,h,w
        # src=src.permute(0,3,1,2)#b,p,h,w
        # # d1=torch.sqrt(torch.mean((d1-d1.mean(dim=1,keepdim=True))**2,dim=1,keepdim=True)) # 深度一致性惩罚,b,1,h,w
        # # src = torch.sqrt(torch.mean((src - src.mean(dim=1, keepdim=True)) ** 2, dim=1, keepdim=True))  # 颜色一致性 b,1,h,w
        # d1_=d1.mean(dim=1,keepdim=True)
        # d2_=src.mean(dim=1,keepdim=True)
        # norm=torch.sqrt(torch.mean((d1-d1_)**2,dim=1,keepdim=True))* torch.sqrt(torch.mean((src - d2_) ** 2, dim=1, keepdim=True))
        # d1=torch.mean((d1-d1_)*(src-d2_),dim=1,keepdim=True) # b,d,h,w
        # m=norm<=0
        # d1=d1/norm
        # d1[m]=threshold # b,1,h,w
        # del d1_,d2_,norm,m
        # prob = (2-prob-d1)/2.0
        # prob=F.conv2d(prob, self.CNN.weight, padding=(self.size - 1) // 2)
        prob = 1 - prob

        return prob, depth, norm_map.squeeze(1).permute(0, 3, 1, 2), mask
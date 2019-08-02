#-*- coding:utf-8 -*-
import torch
import argparse
from  ncc import NCC
from dataset import NccDataSet
from torch.utils.data import DataLoader
import visdom
from torch.nn import DataParallel
import torch.nn.functional as F
import sys
import re
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="Run ncc_optim on depths.")
parser.add_argument("--data_dir",dest="data_dir",default="/home/haibao637/data/tankandtemples/intermediate/Family/") #总数据集目录
vis=visdom.Visdom(env="MVSNet")
args = parser.parse_args()
def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
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

def vis_show(group,*tensors):
	for index,tensor in enumerate(tensors):
		vis.images(tensor,win=group+" %d"%(index))

def main():

	nccnet=NCC(vis,size=3).cuda()
	nccnet=DataParallel(nccnet)
	dataset=NccDataSet(args.data_dir,3)
	dataloader=DataLoader(dataset=dataset,shuffle=False,drop_last=False,batch_size=1,num_workers=1)
	threshold=0.7
	for batch_id,batch in enumerate(dataloader):
		ref_names=batch["ref_name"]
		ref_image=batch["ref_image"].permute(0,3,1,2)
		images=batch["images"].reshape(-1,ref_image.shape[2],ref_image.shape[3],batch["ref_image"].shape[-1]).permute(0,3,1,2)
		images=images.reshape(ref_image.shape[0],-1,ref_image.shape[1],ref_image.shape[2],ref_image.shape[3])#b,n,c,h,w
		depth_map =batch["depth_map"].cuda()
		norm_map = batch["norm_map"].cuda()
		norm_map=F.normalize(norm_map,dim=1)
		Ks=batch["Ks"]
		Rs=batch["Rs"]
		Ts=batch["Ts"]
		# print(ref_names)
		#==============================
		#initialize the reference confidence map(NCC score map) with the score of the current estimates
		# depth_size=torch.ones(ref_image.shape[0])
		ones=torch.ones(ref_image.shape[0])
		pre_prob,pre_depth,pre_norm,pre_mask=nccnet(ref_image, images,Ks,Rs,Ts,
		                                   depth_map,norm_map,dilation=ones,depth_size=ones,threshold=ones*threshold)
		# dst = nccnet(ref_image, images, Ks, Rs, Ts,
		#                                                  depth_map, norm_map, dilation=ones, depth_size=ones,
		#                                                  threshold=ones * threshold)
		# dst=(dst+1)/2.0
		# # print(dst.min(), dst.mean(), dst.max())
		# vis_show("dst12",dst.clamp(0,1.0).cpu())
		# continue
		# print(pre_prob.min(),pre_prob.mean())
		# print(torch.sum(pre_mask))
		pre_prob=pre_prob.detach()
		pre_depth=pre_depth.detach()
		pre_mask=pre_mask.detach()
		maxs = pre_depth.max()* 1.1

		mins= pre_depth.min()  * 0.9 # 扩大范围
		iter_depths = []
		iter_depths.append((255 * (pre_depth - mins) / (maxs - mins)).squeeze(1).cpu())
		#==============================
		final_depth=pre_depth*1.0
		vis_show("initial",ref_image,
		         pre_mask.cpu(),
		         pre_prob.clamp(0, 1.0).cpu(),
		         (255.0 * (final_depth - mins) / (maxs - mins)).detach().cpu(),
		         pre_norm.cpu()
		         )
		scale=0.1
		# return
		# optimizer=torch.optim.Adam([depth_map],lr=1e-3)
		# ground_truth=torch.zeros(final_depth.shape).cuda()
		# criternion=torch.nn.MSELoss()
		for i in range(1,4):
			# 随机更新normal
			# optimizer.zero_grad()
			print("%.2f%% iteration %d ..." % (100.0 * batch_id * dataloader.batch_size / len(dataloader.dataset), i))

			# if i<5:
			# 	iters=1
			# 	ds=3
			# elif i<10:
			# 	iters = 2
			# 	ds=3
			# else:
			# 	scale = 0.5**(1+(i-10)//20)
			# 	iters=1
			# 	ds=3
			iters=1*i
			ds=3
				# depth_map=final_depth*(1+(torch.rand(final_depth.shape).cuda()-0.5)*0.01)
			prob,depth,norm,mask = nccnet(ref_image, images,Ks,Rs,Ts,
		                                   depth_map,norm_map,dilation=ones*iters,depth_size=ones*ds ,threshold=ones*threshold)


			prob=prob.detach()
			depth=depth.detach()
			norm=norm.detach()
			mask=mask.detach()
			print(pre_prob[mask].min(), pre_prob[mask].mean())

			# #======================================
			#

			mask = mask & (prob < pre_prob)
			m = torch.cat([mask] * 3, dim=1)
			pre_norm[m] = norm[m]

			pre_prob[mask] = prob[mask]
			# depth_map[mask] = depth[mask]  # 传播不进行扰
			# if i%10>0:
			norm_map[m]=norm[m]
			depth_map[mask] = depth[mask]  # 传播不进行扰动
			# else:
				# scale*=0.5
				# norm_map[m] = (norm + scale * 0.1 * (2.0 * torch.rand(norm.shape).cuda() - 1.0))[m]
				# depth_map[mask] =( depth  + scale * (2.0*torch.rand(depth.shape).cuda() - 1.0))[mask]
			depth_map[mask==False]=final_depth[mask==False]
			norm_map[m==False]=pre_norm[m==False]


			# depth_map[mask]=depth[mask]
			depth_map=depth_map.clamp(mins,maxs)

			norm_map=F.normalize(norm_map,dim=1)
			# m=norm_map[...,2]<0
			# norm_map[m,:]*=-1
			pre_mask=mask|(pre_mask)
			# mask = (prob <= 2e-1)&(mask)
			final_depth[mask] = depth[mask]

			#======================================
			#visualize
			vis_show("iteration",
			         mask.cpu().type(torch.float32),
			         pre_mask.cpu().type(torch.float32),
			         # (255 * (depth_map - mins) / (maxs - mins)).cpu(),
			         pre_prob.clamp(0,1.0).cpu(),
			         (255.0*(final_depth-mins)/(maxs-mins)).cpu(),
			         pre_norm.cpu())
			del prob,depth,mask
		for index,ref_name in enumerate(ref_names):
			write_pfm(os.path.splitext(ref_name)[0]+"_init.pfm",final_depth[index].detach().squeeze(0).cpu().numpy())
		# return
if __name__=="__main__":
	main()

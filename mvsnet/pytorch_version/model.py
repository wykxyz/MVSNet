import torch.nn as nn
import torch
from pytorch_version.net import FeatureNet

class MVSNet(nn.Module):
    def __init__(self):
        super(MVSNet, self).__init__()
        self.feature=FeatureNet()
    def forward(self,images,cams,depth_image):
        """
        
        :param images: b,v,c,h,w
        :param cams: 
        :param depth_image: 
        :return: 
        """
        batch_size,view_num,channel,height,width=images.shape
        images=torch.reshape(images,[-1,channel,height,width])
        features=self.feature.forward(images).reshape([batch_size,view_num,-1,height,width])
        ref_feature=features[:,0,...]#b,c,h,w
        view_features=features[:,1:,...]#b,v,c,h,w



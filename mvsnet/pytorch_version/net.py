import torch
import torch.nn as nn
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.sequential=torch.nn.Sequential()
    def forward(self,image):
        return self.sequential(image)
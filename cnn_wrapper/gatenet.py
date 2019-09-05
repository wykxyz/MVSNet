from cnn_wrapper.network import Network, layer
import tensorflow as tf
#Gatenet: https://github.com/sczhou/DAVANet/blob/master/models/submodules.py


class Gatenet(Network):

    def setup(self):
        #print ('2D with 32 filters')
        base_filter = 16
        (self.feed('data')
         .conv(3, base_filter, 1, relu=False, name='gate_conv0')
         .leaky_relu(name='gate_leaky_relu')
         .resnet_block(1, base_filter, dilation=[1, 1], name='gate_res_block0')
         .conv(1, 1, 1, relu=False, name='gate_conv1')
         .sigmoid(name='gate_sigmoid'))


class NonlocalNet(Network):
    def setup(self):
        base_filter = 16
        (self.feed('data')
        .non_local(1, base_filter, 1, relu=False, name='nonlocal')
        )

class NonlocalNetForViewNum(Network):
    def setup(self):
        base_filter = 16
        (self.feed('data')
        .non_local_viewdim3d(1, base_filter, relu=False,  bn=False, name='nonlocalviewdim3d')
        )
 
 
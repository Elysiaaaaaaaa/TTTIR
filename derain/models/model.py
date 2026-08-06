
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .layers import *
except ImportError:
    from layers import *
import numpy as np
import cv2

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()

        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True,bias=False,norm=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True,bias=False,norm=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True,bias=False,norm=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()

        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    




class Encoder(nn.Module):
    def __init__(self, input_channel=32, num_blocks=[3,3,3],smooth=False):
        super().__init__()
        
        self.feat_stem =  BasicConv(3, input_channel, kernel_size=3, relu=True, stride=1, bias=False, norm=True)

        self.encoder1 = SESBlock(input_channel,smooth=smooth, depths=num_blocks[0])
        
        self.donwsamp1 = BasicConv(input_channel, input_channel*2, kernel_size=3, relu=True, stride=2, bias=False, norm=True)
        
        self.encoder2 = SESBlock(input_channel * 2,smooth=smooth, depths=num_blocks[1])
        
        self.donwsamp2 = BasicConv(input_channel*2, input_channel*4, kernel_size=3, relu=True, stride=2, bias=False, norm=True)

        self.encoder3 = SESBlock(input_channel * 4,smooth=smooth, depths=num_blocks[2])

        self.scm1 = SCM(input_channel*2)
        self.fam1 = FAM(input_channel*2)
        self.scm2 = SCM(input_channel*4)
        self.fam2 = FAM(input_channel*4)

    def forward(self, x):

        x_2 = F.interpolate(x, scale_factor=0.5)    # 3,H/2
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 3,H/4

        x_2_ = self.scm1(x_2)
        x_4_ = self.scm2(x_4)

        x = self.feat_stem(x)
        x1 = self.encoder1(x)

        x2 = self.donwsamp1(x1)
        x2 = self.fam1(x2, x_2_)
        x2 = self.encoder2(x2)

        x3 = self.donwsamp2(x2)
        x3 = self.fam2(x3, x_4_)
        x3 = self.encoder3(x3)

        return x1, x2, x3
    



class Decoder(nn.Module):
    def __init__(self, input_channel = 32, num_blocks=[3,3,3],smooth=False):
        super().__init__()

        # self.upsamp3 = BasicConv(input_channel*8, input_channel*4, kernel_size=4, relu=True, stride=2, transpose=True, bias=False, norm=True)
        
        self.decoder3 = SESBlock(input_channel * 4,smooth=smooth, depths=num_blocks[2])

        self.upsamp2 = BasicConv(input_channel*4, input_channel*2, kernel_size=4, relu=True, stride=2, transpose=True, bias=False, norm=True)
        self.decoder2 = SESBlock(input_channel * 2,smooth=smooth, depths=num_blocks[1])

        self.upsamp1 = BasicConv(input_channel*2, input_channel, kernel_size=4, relu=True, stride=2, transpose=True, bias=False, norm=True)
        self.decoder1 = SESBlock(input_channel,smooth=smooth, depths=num_blocks[0])


        self.out_conv = nn.ModuleList([
                BasicConv(input_channel * 4, 3, kernel_size=3, relu=False, stride=1, bias=True),
                BasicConv(input_channel * 2 , 3, kernel_size=3, relu=False, stride=1, bias=True),
                BasicConv(input_channel, 3, kernel_size=3, relu=False, stride=1,bias=True)
            ])

        self.Convs = nn.ModuleList([
            BasicConv(input_channel * 4, input_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(input_channel * 2, input_channel, kernel_size=1, relu=True, stride=1),
        ])

    def forward(self, x1, x2, x3, x):

        x_2 = F.interpolate(x, scale_factor=0.5)    # 3,H/2
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 3,H/4
        
        d3 = self.decoder3(x3)
        out3 = self.out_conv[0](d3) + x_4

        d2 = self.upsamp2(d3)
        d2 = self.Convs[0](torch.cat([d2,x2],dim=1))
        d2 = self.decoder2(d2)
        out2 = self.out_conv[1](d2) + x_2

        d1 = self.upsamp1(d2)
        d1 = self.Convs[1](torch.cat([d1,x1],dim=1))
        d1 = self.decoder1(d1)
        out1 = self.out_conv[2](d1) + x

        output = [out3,out2,out1]

        return output



class Model1(nn.Module):
    def __init__(self, num_block=[1,1,1]):
        super(Model1, self).__init__()
        base_channel = 32
        self.Encoder = Encoder(input_channel=base_channel, 
                               num_blocks=num_block)

        self.Decoder = Decoder(input_channel=base_channel,
                               num_blocks=num_block)

    def forward(self, x):

        x1, x2, x3 = self.Encoder(x)
        output = self.Decoder(x1, x2, x3, x)

        return output

def build_net():

    return Model1()

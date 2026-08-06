import torch.nn as nn
import torch
try:
    from models.archs.layers import SESBlock
except Exception:
    from layers import SESBlock
import torch.nn.functional as F

class TTTIR(nn.Module):
    def __init__(self, nc=32, num_block=[1,1,1,1,1]):
        super(TTTIR,self).__init__()
        self.conv0 = nn.Conv2d(3,nc,1,1,0)
        self.conv1 = SESBlock(nc, depths=num_block[0])
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = SESBlock(nc*2, depths=num_block[1])
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = SESBlock(nc*3, depths=num_block[2])
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = SESBlock(nc*2, depths=num_block[3])
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = SESBlock(nc, depths=num_block[4])
        self.convout = nn.Conv2d(nc,3,1,1,0)


    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        xout = x_ori + xout

        return xout
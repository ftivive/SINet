import torch
import torch.nn as nn
from model.conv_fcn import BaseConv, BaseSIConv
from model.convmod import Block

# Upsampling layer
class Interpolate(nn.Module):
    def __init__(self, sc):
       super(Interpolate, self).__init__()
       self.sc     = sc
       self.interp = nn.functional.interpolate
       #
    def forward(self, x):
       return self.interp(x, scale_factor = self.sc, mode='bilinear', align_corners=True)   

# Decoder block
class Dec_Blk(nn.Module):
    def __init__(self, in_chs , out_chs):
       super(Dec_Blk, self).__init__()
       self.dec_blk = nn.Sequential(MixDConv(in_chs, out_chs),
                                    nn.ConvTranspose2d(out_chs, out_chs, kernel_size=3, stride=2, padding=1, output_padding=1)) 
    def forward(self, x):
       return self.dec_blk(x)

class MixDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixDConv, self).__init__()        
        self.pyconv1A = BaseConv(in_channels, in_channels//4, 3, 1, 3, 3, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 8)
        self.pyconv1B = BaseConv(in_channels, in_channels//4, 3, 1, 2, 2, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 4)
        self.pyconv1C = BaseConv(in_channels, in_channels//4, 3, 1, 1, 1, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 1)       
        self.pyconv1D = BaseConv(in_channels, in_channels//4, 1, 1, 0, 1, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 1)          
        self.pyconv1F = BaseSIConv(in_channels, out_channels, 3, 1, 1, 1, activation = nn.ReLU(inplace =  True), use_bn = True)        
    def forward(self, input):
        output = torch.cat([self.pyconv1A(input), self.pyconv1B(input), self.pyconv1C(input), self.pyconv1D(input)], 1)
        return self.pyconv1F(output) 

class Decoder(nn.Module):
    def __init__(self, encoder_channels = (64, 128, 256, 512),
                       att              = False):
        
        super(Decoder, self).__init__()
        #
        self.att = att
        #
        self.p4  = Dec_Blk(encoder_channels[-1], encoder_channels[-2])
        self.b4  = Block(encoder_channels[-2])
        #
        self.p3  = Dec_Blk(encoder_channels[-1], encoder_channels[-3])
        self.b3  = Block(encoder_channels[-3])
        #
        self.p2  = Dec_Blk(encoder_channels[-2], encoder_channels[-4])        
        self.b2  = Block(encoder_channels[-4])
        #
        self.p1  = Dec_Blk(encoder_channels[-3], encoder_channels[-4])
        self.b1  = Block(encoder_channels[-4])
        #
        self.up3 = Interpolate(4)
        self.up2 = Interpolate(2)
        #
        self.init_weight()


    def forward(self, res, att_map):
        if self.att:           
            x    = self.b4(self.p4(res[3]))
            x    = self.p3(torch.cat([x,    res[2]],1))   
            out0 = self.b3(x) 
            x    = self.p2(torch.cat([out0, res[1]],1))    
            out1 = self.b2(x)
            x    = self.p1(torch.cat([out1, res[0]],1))   
            out2 = self.b1(x)
            #           
            return out2, out1, out0
        else:
            x  = self.b4(self.p4(res[3]))
            #
            x  = self.p3(torch.cat([x, res[2]],1)) * att_map[0]
            x  = self.b3(x)
            x1 = self.up3(x)
            #
            x  = self.p2(torch.cat([x, res[1]],1)) * att_map[1]
            x  = self.b2(x) 
            x2 = self.up2(x)
            # 
            x  = self.p1(torch.cat([x, res[0]],1)) * att_map[2]
            x  = self.b1(x)            
            #
            return torch.cat([x, x1, x2],1)
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


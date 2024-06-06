import torch
import torch.nn as nn
from model.convmod import ATT_BK
from utils.utils import BaseConv, BaseSIConv

# Upsampling layer
class Interpolate(nn.Module):
    def __init__(self, sc):
       super(Interpolate, self).__init__()
       self.sc     = sc
       self.interp = nn.functional.interpolate
       #
    def forward(self, x):
       return self.interp(x, scale_factor = self.sc, mode='bilinear', align_corners=True)    

# Decoder block (DEC-BK)
class DEC_BK(nn.Module):
    def __init__(self, in_chs , out_chs):
        super(DEC_BK, self).__init__()
        self.pyconv1A = BaseConv(in_chs, in_chs//4, 3, 1, 3, 3, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 8)
        self.pyconv1B = BaseConv(in_chs, in_chs//4, 3, 1, 2, 2, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 4)
        self.pyconv1C = BaseConv(in_chs, in_chs//4, 3, 1, 1, 1, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 1)       
        self.pyconv1D = BaseConv(in_chs, in_chs//4, 1, 1, 0, 1, activation = nn.ReLU(inplace =  True), use_bn = False, groups = 1)          
        self.pyconv1F = BaseSIConv(in_chs, out_chs, 3, 1, 1, 1, activation = nn.ReLU(inplace =  True), use_bn = True)        
        self.upsample = nn.ConvTranspose2d(out_chs, out_chs, kernel_size=3, stride=2, padding=1, output_padding=1)
       
    def forward(self, x):
        output = torch.cat([self.pyconv1A(x), self.pyconv1B(x), self.pyconv1C(x), self.pyconv1D(x)], 1)
        output = self.pyconv1F(output)        
        return self.upsample(output)

# Decoder Module
class Decoder(nn.Module):
    def __init__(self, encoder_channels = (64, 128, 256, 512),
                       att              = False):
        
        super(Decoder, self).__init__()
        #
        self.att = att
        #
        self.p4  = DEC_BK(encoder_channels[-1], encoder_channels[-2])
        self.b4  = ATT_BK(encoder_channels[-2])
        #
        self.p3  = DEC_BK(encoder_channels[-1], encoder_channels[-3])
        self.b3  = ATT_BK(encoder_channels[-3])
        #
        self.p2  = DEC_BK(encoder_channels[-2], encoder_channels[-4])        
        self.b2  = ATT_BK(encoder_channels[-4])
        #
        self.p1  = DEC_BK(encoder_channels[-3], encoder_channels[-4])
        self.b1  = ATT_BK(encoder_channels[-4])
        #
        self.up3 = Interpolate(4)
        self.up2 = Interpolate(2)

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
    



import torch
from torch import nn
from model.conv_fcn import BaseConv, BaseSIConv
from model.dec_module import Decoder


class SINet(nn.Module):
    def __init__(self, in_fs, num_ouputs):
        super(SINet, self).__init__()        
        self.fext = FEN(3, in_fs)        
        self.amp  = Att_Decoder(in_fs, num_ouputs)        
        self.dmp  = Dmp_Decoder(in_fs, num_ouputs)        
        self._random_init_weights()

    def _random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)        

    def forward(self, input):
        input   = self.fext(input)
        amp_out = self.amp(input)
        dmp_out = self.dmp(input, amp_out)      
        return dmp_out, amp_out


class Enc_Blk(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Enc_Blk, self).__init__()
        sub_chs  = out_chs // 4
        self.cv0 = BaseSIConv(in_chs,      sub_chs, 1, 1, 0, 1, activation = None,                  use_bn = True)
        self.cv1 = BaseSIConv(in_chs,      sub_chs, 3, 1, 1, 1, activation = None,                  use_bn = True)
        self.cv2 = BaseSIConv(2*sub_chs, 2*sub_chs, 5, 1, 2, 1, activation = None,                  use_bn = True)
        self.cv3 = BaseSIConv(4*sub_chs, 4*sub_chs, 7, 1, 3, 1, activation = None,                  use_bn = True)
        self.cv4 = BaseConv(sub_chs * 8,   out_chs, 1, 1, 0, 1, activation = nn.ReLU(inplace=True), use_bn = True)

    def forward(self, x):
        #  1
        x0_raw = self.cv0(x)
        x1_raw = self.cv1(x)
        #  2                           1       1
        x2_raw = self.cv2(torch.cat([x0_raw, x1_raw], 1))
        #  4                           1       1       2
        x3_raw = self.cv3(torch.cat([x0_raw, x1_raw, x2_raw], 1))
        #  8                           1         1       2      4
        return   self.cv4(torch.cat([x0_raw, x1_raw, x2_raw, x3_raw], 1))         

class FEN(nn.Module):
    def __init__(self, in_chs, in_fs):
        super(FEN, self).__init__()
        self.Enconv1 = nn.Sequential(Enc_Blk(3,         in_fs), nn.MaxPool2d(2, 2))
        self.Enconv2 = nn.Sequential(MixConv(in_fs,   2*in_fs), nn.MaxPool2d(2, 2))
        self.Enconv3 = nn.Sequential(MixConv(2*in_fs, 4*in_fs), nn.MaxPool2d(2, 2))
        self.Enconv4 = nn.Sequential(MixConv(4*in_fs, 8*in_fs), nn.MaxPool2d(2, 2))
        self.Enconv5 = nn.Sequential(BaseConv(15*in_fs, 16*in_fs, 3, 1, 1, 1, activation = nn.ReLU(inplace=True), use_bn = True), nn.MaxPool2d(2, 2))  
        #
        self.dp0     = nn.AvgPool2d(8, 8) 
        self.dp1     = nn.AvgPool2d(4, 4) 
        self.dp2     = nn.AvgPool2d(2, 2)  
        # 

    def forward(self, input):   
        dnout1_2 = self.Enconv1(input)           
        dnout2_2 = self.Enconv2(dnout1_2)           
        dnout3_2 = self.Enconv3(dnout2_2)
        dnout4_2 = self.Enconv4(dnout3_2)
        input    = self.Enconv5(torch.cat([dnout4_2, self.dp2(dnout3_2), self.dp1(dnout2_2), self.dp0(dnout1_2)],1))    
        return [dnout2_2, dnout3_2, dnout4_2, input]
    

class Att_Decoder(nn.Module):
    def __init__(self, in_fs, outputs):
        super(Att_Decoder, self).__init__()        
        #
        self.decoder = Decoder((2*in_fs, 4*in_fs, 8*in_fs, 16*in_fs), True) 
        self.att_0   = BaseConv(4*in_fs, outputs, 3, 1, 1, 1, activation = nn.Sigmoid(), use_bn = True)
        self.att_1   = BaseConv(2*in_fs, outputs, 3, 1, 1, 1, activation = nn.Sigmoid(), use_bn = True)
        self.att_2   = BaseConv(2*in_fs, outputs, 3, 1, 1, 1, activation = nn.Sigmoid(), use_bn = True)
       
    def forward(self, input):   		
        out2, out1, out0 = self.decoder(input, [])
        out0             = self.att_0(out0)
        out1             = self.att_1(out1)
        out2             = self.att_2(out2)
        return [out0, out1, out2]


class weighted_fusion(nn.Module):
    def __init__(self, channels):
        super(weighted_fusion, self).__init__()
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.weight(self.gap(x).view(b,c))
        return nn.functional.softmax(y, dim=1)


class Dmp_Decoder(nn.Module):
    def __init__(self, in_fs, outputs):
        super(Dmp_Decoder, self).__init__()        
        self.decoder  = Decoder((2*in_fs, 4*in_fs, 8*in_fs, 16*in_fs), False)
        self.conv_F1  = BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation = nn.ReLU(inplace=True), use_bn = True)    
        self.conv_F2  = BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation = nn.ReLU(inplace=True), use_bn = True)    
        self.conv_F3  = BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation = nn.ReLU(inplace=True), use_bn = True)    
        #        
        self.out_O1   = nn.Sequential(BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation =  nn.ReLU(inplace = True), use_bn = True), 
                                      BaseConv(8*in_fs, outputs, 1, 1, 0, 1, activation = nn.ReLU6(inplace = True), use_bn = False))

        self.out_O2   = nn.Sequential(BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation =  nn.ReLU(inplace = True), use_bn = True),
                                      BaseConv(8*in_fs, outputs, 1, 1, 0, 1, activation = nn.ReLU6(inplace = True), use_bn = False))

        self.out_O3   = nn.Sequential(BaseConv(8*in_fs, 8*in_fs, 3, 1, 1, 1, activation =  nn.ReLU(inplace = True), use_bn = True),
                                      BaseConv(8*in_fs, outputs, 1, 1, 0, 1, activation = nn.ReLU6(inplace = True), use_bn = False))

        self.weights  = weighted_fusion(3)
        #   

    def forward(self, input, att_map):   		        
        output    = self.decoder(input, att_map)                

        output    = self.conv_F1(output)
        dmp_LR    =  self.out_O1(output) 

        output    = self.conv_F2(output)
        dmp_MR    =  self.out_O2(output) 

        output    = self.conv_F3(output)
        dmp_HR    =  self.out_O3(output)
        #
        weight    = self.weights(torch.cat([dmp_HR, dmp_MR, dmp_LR],1))

        dmp_HR    = weight[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * dmp_HR + weight[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * dmp_MR + weight[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3) * dmp_LR 
        
        return [dmp_HR, dmp_MR, dmp_LR]

class MixConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixConv, self).__init__()
        self.conv1A = nn.Sequential(BaseConv(in_channels,  out_channels, 1, 1, 0, 1,   activation = nn.ReLU(inplace = True), use_bn = True),
                                    BaseSIConv(out_channels, out_channels, 3, 1, 1, 1, activation = nn.ReLU(inplace = True), use_bn = True))  

        self.conv1B = nn.Sequential(BaseConv(in_channels,  out_channels, 3, 1, 1, 1,   activation = nn.ReLU(inplace = True), use_bn = True),
                                    BaseSIConv(out_channels, out_channels, 3, 1, 1, 1, activation = nn.ReLU(inplace = True), use_bn = True))  

        self.conv1C = BaseConv(out_channels * 2, out_channels, 1, 1, 0, 1, activation = nn.ReLU(inplace = True), use_bn = True)
         
    def forward(self, input):
        return self.conv1C(torch.cat([self.conv1A(input), self.conv1B(input)], 1)) 
 

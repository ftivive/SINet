import torch
import torch.nn as nn
from model.siconv2d import SIConv2d

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 1, dilation = 1, activation=None, use_bn=False, groups = 1):
        super(BaseConv, self).__init__()
        self.use_bn     = use_bn
        self.activation = activation
        if self.use_bn:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding=padding, dilation=dilation, bias=False, groups = groups)
            self.bn   = nn.BatchNorm2d(out_channels)	
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding=padding, dilation=dilation, bias=True, groups = groups)
    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)
        return input

class BaseSIConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 1, dilation = 1, activation=None, use_bn=False, groups = 1):
        super(BaseSIConv, self).__init__()
        self.use_bn     = use_bn        
        self.conv       = SIConv2d(in_channels, out_channels, kernel_size, kernel_size, stride, padding, dilation, groups)
        self.bn         = nn.BatchNorm2d(out_channels)
        self.activation = activation
    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input) 
        if self.activation:   
            input = self.activation(input)
        return input



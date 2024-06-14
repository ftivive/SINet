"""
Created on Thu Jun 25 10:17:42 2020

@author: Steve
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from scipy.ndimage.filters import gaussian_filter
import numpy as np



class SIConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernel_size, den_kernel_size, stride=1, padding=0, dilation=1, groups = 1):
        #
        super(SIConv2d, self).__init__()
        self.num_kernel_size = num_kernel_size
        self.den_kernel_size = den_kernel_size               
        self.out_channels    = out_channels
        self.num_dilation    = dilation
        self.den_dilation    = dilation
        self.num_padding     = padding
        self.den_padding     = padding
        self.stride          = stride
        self.groups          = groups
        self.in_channels     = in_channels
        self.device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       
        self.num_fcn         = nn.Conv2d(in_channels, out_channels, num_kernel_size, stride = stride, padding=self.num_padding, dilation=dilation, bias=True, groups = groups)
        self.den_fcn         = nn.Conv2d(in_channels, out_channels, den_kernel_size, stride = stride, padding=self.den_padding, dilation=dilation, bias=True, groups = groups)
        self.bias_A          = Parameter(torch.empty(out_channels, device=self.device))
        self.bias_A_act_fcn  = nn.Hardtanh(min_val = 0.01, max_val = 1) 
        self.act_fcn         = nn.Softplus()
        #
        self.reset_parameters()
        #
    # Initialize the parameters
    def reset_parameters(self):       
        torch.nn.init.xavier_uniform_(self.num_fcn.weight)   
        #        
        nn.init.uniform_(self.bias_A,      0.5, 1)   # bias A init
        nn.init.normal_(self.num_fcn.bias, 0.01)   # bias B init
        nn.init.normal_(self.den_fcn.bias, 0.01)   # bias C init
        #
        Sz                       = self.den_kernel_size
        in_ch                    = self.in_channels
        out_ch                   = self.out_channels
        n                        = np.zeros((Sz,Sz))
        n[Sz//2,Sz//2]           = 1
        gaussian_2D              = gaussian_filter(n, sigma = Sz/6)
        gaussian_2D              = gaussian_2D / np.sum(gaussian_2D)
        gaussian_2D              = np.repeat(gaussian_2D[np.newaxis, :, :], in_ch//self.groups,  axis=0)  
        gaussian_2D              = np.repeat(gaussian_2D[np.newaxis, :, :], out_ch, axis=0)           
        gaussian_2D              = torch.from_numpy(gaussian_2D).type(torch.float32)           
        self.den_fcn.weight.data = gaussian_2D.to(self.device)

    # In[3]: Define actual computations
    def forward(self, input):
        # Shunting Inhibitory Neuron Computation
        return torch.div(self.num_fcn(input), self.act_fcn(self.den_fcn(input)) + self.bias_A_act_fcn(self.bias_A).unsqueeze(0).unsqueeze(2).unsqueeze(3)) 
	

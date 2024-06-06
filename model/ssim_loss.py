# -*- coding: utf-8 -*-
"""
multi-scale structure similarity index measure
"""
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class ms_ssimloss(torch.nn.Module):
    def __init__(self, in_channels, gauss_filter, fsize, size_average = True):
        super(ms_ssimloss, self).__init__()
        self.ssim_loss = []
        self.level     = len(fsize)        
        for i in range(self.level):
             self.ssim_loss.append(ssim_loss(in_channels, gauss_filter[i], fsize[i], size_average))

    def forward(self, img1, img2):
        avg_loss = 0.0
        for i in range(self.level):           
            avg_loss += self.ssim_loss[i](img1, img2)
            img1      = F.avg_pool2d(img1, (2, 2))
            img2      = F.avg_pool2d(img2, (2, 2))
        return avg_loss



class ssim_loss(_Loss):
    def __init__(self, in_channels, gauss_filter, window_size = 11, size_average = True, dilation = 1):
        super(ssim_loss, self).__init__()
        self.in_channels  = in_channels
        self.size         = window_size
        self.size_average = size_average
        self.dilation     = dilation
        self.padding      = window_size + (dilation - 1) * (window_size - 1) - 1
        self.weight       = gauss_filter

    def forward(self, img1, img2):
        mean1    = F.conv2d(img1, self.weight, padding = self.padding//2, dilation = self.dilation, groups = self.in_channels)
        mean2    = F.conv2d(img2, self.weight, padding = self.padding//2, dilation = self.dilation, groups = self.in_channels)
        mean1_sq = mean1 * mean1
        mean2_sq = mean2 * mean2
        mean_12  = mean1 * mean2

        sigma1_sq = F.conv2d(img1 * img1, self.weight, padding = self.padding//2, dilation = self.dilation, groups = self.in_channels) - mean1_sq 
        sigma2_sq = F.conv2d(img2 * img2, self.weight, padding = self.padding//2, dilation = self.dilation, groups = self.in_channels) - mean2_sq
        sigma_12  = F.conv2d(img1 * img2, self.weight, padding = self.padding//2, dilation = self.dilation, groups = self.in_channels) - mean_12

        C1        = 0.01 ** 2
        C2        = 0.03 ** 2

        ssim      = ((2 * mean_12 + C1) * (2 * sigma_12 + C2)) / ((mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            #out = 1.0 - ssim.mean(3).mean(2).squeeze(1).sum(0)
             out = 1.0 - ssim.mean()
        else:
            out = 1.0 - ssim.view(ssim.size(0), -1).mean(1)
        return out

# -*- coding: utf-8 -*-
"""
@author: tivive
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from model.siconv2d import SIConv2d
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())


# perceptron layer
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

# shunting inhibitory neuron layer
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


# CUDA settings
def setup_cuda():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setting seeds (optional)
    seed                           = 0
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return device


# Train the model over a single epoch
def train_model(model, device, optimizer, denloss, bceloss, l1loss, train_loader, batch_size, num_acc, epoch):
    model.train()
    train_loss = 0.0   
    train_mae  = 0.0
    pbar       = tqdm(train_loader, ncols = 40, desc = 'Training')    
    # Set the gradients to zero before starting backpropagation
    optimizer.zero_grad()
    num_acc = num_acc
    for i, (images, density_HR, density_MR, density_LR, att, gt_count) in enumerate(pbar):        
        images     = images.to(device)
        density_HR = density_HR.to(device)
        density_MR = density_MR.to(device)
        density_LR = density_LR.to(device)
        att        = att.to(device)
        gt_count   = gt_count.to(device)        

        # Perform a forward pass
        den_map, attention = model(images)             

        # Calculate the training loss
        # multi scale SSIM loss
        loss_dmp    = denloss(den_map[0], density_HR)
        # count loss
        loss_count  = l1loss(den_map[0].sum(1).sum(1).sum(1), gt_count)   
        # binary loss for segmentation
        loss_att = 0.0
        for amp in attention:
          amp_gt = att	
          # Downsample the ground-truth attention map to the output size
          if amp_gt.shape[2:] != amp.shape[2:]:
             amp_gt = F.interpolate(amp_gt, amp.shape[2:], mode='nearest')
          loss_att += bceloss(amp, amp_gt) 
        #
        loss        = loss_att * 0.1 + loss_dmp + loss_count
        loss        = loss / (num_acc * batch_size) 		
        train_loss += loss.item()
        train_mae  += (torch.abs(den_map[0].sum(1).sum(1).sum(1) - gt_count)).sum().item()  
     
        # Compute gradient of the loss fn w.r.t the trainable weights
        # Perform gradient accumulation        
        loss.backward()
        if ((i + 1) % num_acc == 0) or (i + 1 == len(train_loader)):
            # Updates the trainable weights
            optimizer.step()
            # Set the gradients to zero before starting backpropagation
            optimizer.zero_grad()	
    return train_loss / len(train_loader), train_mae / len(train_loader)

# Validate the model over a single epoch
def validate_model(model, device, valid_loader):
    model.eval()
    with torch.no_grad():
        mae, mse = 0.0, 0.0
        for iteration, (images, gt) in enumerate(valid_loader):
            images  = images.to(device)   
            gt      = gt.to(device)   
            # Perform a forward pass
            outputs, _ = model(images)  
            mae += torch.abs(outputs[0].sum() - gt).item()
            mse += ((outputs[0].sum() - gt) ** 2).item()
        mae /= len(valid_loader)
        mse /= len(valid_loader)
        mse  = mse ** 0.5            
    return mae, mse

import torch
import torch.nn as nn
from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(dim)  
        self.fc1  = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos  = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2  = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act  = nn.ReLU(inplace = True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)	
        self.a    = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(dim, dim, 5, padding=2, groups=dim))
        self.v    = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        

    def forward(self, x):
        B, C, H, W = x.shape        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x

class ATT_BK(nn.Module):
    def __init__(self, dim, mlp_ratio = 4, drop_path = 0):
        super().__init__()
      
        self.attn              = ConvMod(dim)
        self.mlp               = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1     = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2     = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path         = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


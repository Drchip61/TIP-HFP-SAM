import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic
import torch.fft


#b,32,32,768
#test input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_feature_channels_colormap(fm, save_prefix="feature", colormap='jet'):
    fm = fm.cpu()
    num_show = 6
    for i in range(num_show):
        fmap = fm[i]
        fmap -= fmap.min()
        fmap /= fmap.max()
        fmap_resized = TF.resize(fmap.unsqueeze(0), [512, 512])[0]

        # 应用伪彩色映射
        cmap = cm.get_cmap(colormap)
        colored_map = cmap(fmap_resized.numpy())  # 返回 RGBA
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)  # 取 RGB 且转 uint8

        img = Image.fromarray(colored_map)
        img.save(f"{save_prefix}_channel_{i}.png")

        # 同时显示
        plt.subplot(1, num_show, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

#show_feature_channels_colormap(feature_map)

class senet(nn.Module):
    def __init__(self,c=768,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out+max_out
        x = x*self.sigmoid(out)
        #x = x.view(b,c,h,w)
        return x+res

class QuickGELU(nn.Module):
    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(1.702*x)
class fre_adapter(nn.Module):
    def __init__(self,c=768,r=12):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        self.fc2 = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        self.IN = nn.LayerNorm(c)
        self.init_weights()
        #self.reduce = nn.Sequential(nn.Linear(2*c,c),QuickGELU())
        #self.se = senet()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6) 
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)
    
    def forward(self,x,fre_mask):
        ori = x
        #print(fre_mask.size())
        # 设备无关：跟随输入特征所在设备（cuda/cpu/mps）
        fre_mask_expand = fre_mask.unsqueeze(1).expand(-1,768,-1,-1).to(x.device)
        x_change = x.permute(0,3,1,2)
        mask_x = x_change * fre_mask_expand
        #mask_x = mask_x[0]
        #show_feature_channels_colormap(mask_x)
        x_fre = mask_x.permute(0,2,3,1) 
        #print(x.size())
        #x_fft = torch.fft.fft2(x)#.real#.astype(float)
        #x_fft = x_fft.real.float()
        #print(x_fft.size())
        b,h,w,c = x.size()

        out1 = self.fc1(self.IN(x.view(b,h*w,c))).view(b,h,w,c)
        out2 = self.fc2(self.IN(x_fre.view(b,h*w,c))).view(b,h,w,c)
        #.real
        #real_part = x_fft.reshape(b,h*w, c)
        #print(real_part.size())
        #processed_real = self.linear(real_part)
        #processed_real = self.fc1(self.IN(real_part))
        # 假设处理后的实部大小与原始实部相同
        #processed_fft = processed_real.view(x_fft.shape)
        #processed_fft = torch.complex(processed_real.view(x_fft.shape), torch.zeros_like(processed_real).view(x_fft.shape))
        
        # 步骤 3: 频域回到空域
        #x_ifft = torch.fft.ifft2(processed_fft)
        #x_ifft = x_ifft.real.float()
        #out = self.fc(x_fft.view(b,h*w,c))
        #out2 = x_ifft.view(b,h,w,c)
        return out1+out2+ori

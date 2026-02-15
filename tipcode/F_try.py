import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pywt
import math
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms
import torchvision
import cv2

class FrequencyIndex(nn.Module):
    def __init__(self, keep=True):
        super().__init__()
        self.DWT = DWTForward(J=3

, wave='haar', mode='zero')#.cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero')#.cuda()
        self.gassian = transforms.GaussianBlur(kernel_size=5, sigma=0.1)#.cuda()
        self.laplace = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])#.cuda()
        self.keep = keep
        self.window_size = 16


    def forward(self, x,  pattern='a', mode=None, writer=None, step=None):
        Ylx, Yhx = self.DWT(x)
        #print(Ylx.size(),Yhx[0].size())
        res_ = Yhx[0][0][:,-3,:,:]+Yhx[0][0][:,-2,:,:]+Yhx[0][0][:,-1,:,:]
        res_ = res_/3
        res = res_[0]+res_[1]+res_[2]
        #res = res.permute(1,2,0)
        res = res/3
        res1 = np.array(res)
        #print(res[10:25,10:25])
        # print(res1.shape)
        #plt.imshow(np.array(x[0][0]),cmap = 'bwr')
        plt.imshow(res1,cmap = 'bwr')
        plt.show()
        cv2.imwrite('output.jpg',res1)
        #torchvision.utils.save_image(res, 'test_Fre.png') 
        #torchvision.utils.save_image(Yhx[0][0][:,-3,:,:], 'test_Fre_lh.png')    
        #torchvision.utils.save_image(Yhx[0][0][:,-2,:,:], 'test_Fre_hl.png')  
        #torchvision.utils.save_image(Yhx[0][0][:,-1,:,:], 'test_Fre_hh.png')      
        #print(Ylx.shape,Yhx[0][0][:,-1,:,:].shape,Yhx[1].shape,Yhx[2].shape,Yhx[3].shape)
        Inverse = self.IDWT((Ylx, Yhx))
        

        return Inverse

img = cv2.imread('MAS_Arthropod_Crab_Cam_410.jpg')
#img = cv2.resize(img,(512,512))
a = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
print(a.size())
model = FrequencyIndex(True)
res = model(a)

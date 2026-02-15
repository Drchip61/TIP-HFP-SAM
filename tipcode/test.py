import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F

from dataset import TestDataset

import tqdm


import dataset_fre
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

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

from sam_lora_image_encoder import LoRA_Sam


sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')#"sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()

# pretrain ="SAM-512-fre-20.pth"
# model.load_lora_parameters(pretrain)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",default='checkpoint/SAM-512-fre_final-20.pth', type=str,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", default='Kvasir/image/',type=str,
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path",default='Kvasir/masks/', type=str,
                    help="path to the mask files for testing")
parser.add_argument("--test_fre_path",default='Kvasir/Frequency_2/', type=str,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", default='test_samb_kvasir',type=str,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path,args.test_fre_path, 512)
model = model
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt,fre, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        fre = fre.to(device)

        res0,res1,res2= model(image,fre,1,512)
        res = res1#[0][1].unsqueeze(0).unsqueeze(0)
        #print(res.size())
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

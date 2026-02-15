import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset


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

pretrain ="sam_vit_b_01ec64.pth"
model.load_lora_parameters(pretrain)

model = model.train()


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path",default='sam2_hiera_base_plus.pt', type=str,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", default='train/Image/',type=str,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", default='train/Masks/', type=str,
                    help="path to the mask file for training")
parser.add_argument("--train_fre_path", default='train/Frequency_2/', type=str,
                    help="path to the mask file for training")
parser.add_argument('--save_path', default='checkpoint',type=str,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce+wiou).mean()

def adjust_learning_rate(optimizer,epoch,start_lr):
    if epoch%20 == 0:  #epoch != 0 and
    #lr = start_lr*(1-epoch/EPOCH)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]*0.1
        print(param_group["lr"])

def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path,args.train_fre_path, 512, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    #model = SAM2UNet(args.hiera_path)
    #model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        # adjust_learning_rate(optim, epoch, 0.01)
        # print('LR is:', optim.state_dict()['param_groups'][0]['lr'])
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target1 = batch['label1']
            target2 = batch['label2']
            fre = batch['fre']
            x = x.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            fre = fre.to(device)
            fre = fre.squeeze(1)
            #print(target.shape,fre.shape)
            optim.zero_grad()
            pred0,pred1,pred2=  model(x,fre,1,512)#model(x,fre)
            loss0 = structure_loss(pred0, target1)
            loss1 = structure_loss(pred1, target1)
            # loss2 = structure_loss(pred2, target2)
            loss = loss0+loss1#+loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path,'SAM-512-fps-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM-512-fps-%d.pth'% (epoch + 1)))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
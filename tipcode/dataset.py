import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label1,label2,fre = data['image'], data['label1'],data['label2'],data['fre']
        return {'image': F.to_tensor(image), 'label1': F.to_tensor(label1),'label2':F.to_tensor(label2),'fre': F.to_tensor(fre)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label1,label2,fre = data['image'], data['label1'],data['label2'],data['fre']

        return {'image': F.resize(image, self.size), 'label1': F.resize(label1, (128,128), interpolation=InterpolationMode.BICUBIC), 'label2': F.resize(label2,self.size, interpolation=InterpolationMode.BICUBIC), 'fre': F.resize(fre, self.size)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label1, label2, fre = data['image'], data['label1'], data['label2'], data['fre']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label1': F.hflip(label1),'label2':F.hflip(label2), 'fre': F.hflip(fre)}

        return {'image': image, 'label1': label1,'label2':label2,'fre': fre}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label1, label2, fre = data['image'], data['label1'], data['label2'], data['fre']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label1': F.vflip(label1),'label2':F.vflip(label2), 'fre': F.vflip(fre)}

        return {'image': image, 'label1': label1,'label2':label2, 'fre': fre}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, label1, label2, fre = data['image'], data['label1'], data['label2'], data['fre']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label1': label1,'label2':label2,'fre':fre}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, fre_root,size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.fre = [fre_root+f for f in os.listdir(fre_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fre = sorted(self.fre)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        fre = self.binary_loader(self.fre[idx])
        data = {'image': image, 'label1': label,'label2':label,'fre':fre}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root,fre_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.fre = [fre_root + f for f in os.listdir(fre_root) if f.endswith('.jpg') ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fre = sorted(self.fre)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.transform_fre = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        fre = self.binary_loader(self.fre[self.index])
        fre = self.transform_fre(fre)#.unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1

        return image, gt,fre, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
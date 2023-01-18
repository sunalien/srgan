# -*- coding = utf-8 -*-
# @Time : 2023/1/18 21:42
# @Author : 加加
# @File : test_dtloader.py
# @Software : PyCharm
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data.dataset import Dataset
from os import listdir
from os.path import join
import os

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


workers = 0
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# 图片通道数
nc = 3
# Size of z latent vector (i.e. size of generator input)
# nz = 100
nz = 3
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0002  # 优化器学习率
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
RB_nums = 5

### 读入数据集
# transformer可以对数据集进行处理：https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
# dset.ImageFolder这个函数的dataroot需要图片在一个子目录下
# 之后可以尝试一些
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def train_transform():
    return Compose([
        ToTensor(),
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TrainDatasetFromFolder_lr(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder_lr, self).__init__()
        self.lrimage_filenames = [join(join(dataset_dir,'/input'), x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.img_transform = train_transform()

    def __getitem__(self, index):
        lr_image = self.img_transform(Image.open(self.lrimage_filenames[index]))
        return lr_image
# TODO:is this necessary
    def __len__(self):
        return len(self.lrimage_filenames)
class TrainDatasetFromFolder_hr(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder_hr, self).__init__()
        self.hrimage_filenames = [join(join(dataset_dir,'/target'), x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.img_transform = train_transform()

    def __getitem__(self, index):
        hr_image = self.img_transform(Image.open(self.hrimage_filenames[index]))
        return hr_image
# TODO:is this necessary
    def __len__(self):
        return len(self.lrimage_filenames)
dataroot = "E:/learn/others/prof.JW_summer_intern/summerintern/super-resolution_radar/dataset/radar_grid_2/train/"

train_set_lr = TrainDatasetFromFolder_lr(dataroot)
train_set_hr = TrainDatasetFromFolder_lr(dataroot)
# val_set = ValDatasetFromFolder('')
trainlr_loader = DataLoader(dataset=train_set_lr, num_workers=0, batch_size=64, shuffle=False)
trainhr_loader = DataLoader(dataset=train_set_hr, num_workers=0, batch_size=64, shuffle=False)
# val_loader = DataLoader(dataset=val_set, num_workers=, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


real_batch = next(iter(trainlr_loader))

# test loading data step
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

real_batch = next(iter(trainhr_loader))

# test loading data step
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

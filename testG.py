from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

### 设置随机种子
# # Set random seed for reproducibility
# manualSeed = 999
# #manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

### 参数设置
dataroot = "E:/learn/others/prof.JW_summer_intern/summerintern/super-resolution_radar/dataset/radar_grid_2/train/input"

# Number of workers for dataloader
# when setting >0, error:
# 在linux系统中可以使用多个子进程加载数据，而在windows系统中不能。所以在windows中要将DataLoader中的num_workers设置为0或者采用默认为0的设置
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
lr = 0.0002 # 优化器学习率
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
RB_nums = 5
scale_factor = 1

### 读入数据集
# transformer可以对数据集进行处理：https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
# dset.ImageFolder这个函数的dataroot需要图片在一个子目录下
# 之后可以尝试一些
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               # TODO: img will lose information when resize&CenterCrop--so we do not resize&CenterCrop in out experiment, right?
                               # transforms.Resize(image_size),
                               # transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))

# # test loading data step
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

### 权重初始化函数
# TODO:m是什么(model??!)
def weights_init(m):
    classname = m.__class__.__name__
    # 一般在查找等算法，没得到预期结果的时候都会输出-1作为一个negative的值
    # 所以这里是指：如果有找到Conv这个字符串
    if classname.find('Conv') != -1:
        # torch.nn.init.uniform_(tensor, a=0.0, b=1.0):N(mean,std^2)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # torch.nn.init.constant_(tensor, val):Fills the input Tensor with the value valval.
        nn.init.constant_(m.bias.data, 0)


# TODO:还没有padding-related to the final img size(SR size)
class Generator(nn.Module):
    def __init__(self,ngpu):
        # TODO:还没搞懂super的用法-https://blog.csdn.net/weixin_44878336/article/details/124658574
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.block1 = nn.Sequential(
            # nz=Size of z latent vector 输入生成器的随机向量
            # 但是SR任务中输入生成器的是LR图片（3*imageSize）,所以in_channels=通道数=3
            # channel:3-->64,kernel=9*9,
            nn.Conv2d(in_channels=nz, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        self.block2 = nn.Sequential(
            (Residual_block(64) for _ in range(RB_nums)),  # TODO: RB_nums需不需要global一下
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
        )

        #
        # 所以这个pixelshuffle是用来放大图片的----所以之前的conv都要保证图片大小不变
        self.block3 = nn.Sequential(
            nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
            # TODO: In our task, we delete this step or ignore the whole block?--
            # nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            # TODO: In our task, we delete this step or ignore the whole block? or remain activation/ layer
            #  --shoud I go through corresponding paper first?
            # nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

        self.block5 = nn.Conv2d(256,3,kernel_size=3,stride=1,padding=1)

    def forward(self,input):
        input1 = self.block1(input)
        output = self.block2(input1)+input1
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        return output


class Residual_block(nn.Module):
    def __init__(self, channels):
        # TODO:还没搞懂super的用法-https://blog.csdn.net/weixin_44878336/article/details/124658574
        super(Residual_block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channels), # BN之后维度是不变的，把channels作为一个向量的维度做BN
            nn.PReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, input):
        return self.main(input)+input


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)


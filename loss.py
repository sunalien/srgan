# -*- coding = utf-8 -*-
# @Time : 2023/1/15 21:58
# @Author : 加加
# @File : loss.py
# @Software : PyCharm
import torch
from torch import nn
from torchvision.models.vgg import vgg16

class G_loss(nn.Module):
    def __init__(self):
        super(G_loss,self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()

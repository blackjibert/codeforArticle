import os
import ssl
import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
#
# cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
#
# class VGG(nn.Module):
#     def __init__(self, net_name):
#         super(VGG, self).__init__()
#
#         # 构建网络的卷积层和池化层，最终输出命名features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
#         self.features = self._make_layers(cfg[net_name])
#
#         # 构建卷积层之后的全连接层以及分类器
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),  # fc1
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),  # fc2
#             nn.ReLU(True),
#             nn.Linear(512, 10),  # fc3，最终cifar10的输出是10类
#         )
#         # 初始化权重
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
#         return x
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 # layers += [conv2d, nn.ReLU(inplace=True)]
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(v),
#                            nn.ReLU(inplace=True)]
#                 in_channels = v
#         return nn.Sequential(*layers)
#
#
# net = VGG('VGG16')
#
# print(net)

# 定义VGG块
def vgg_block(num_convs, in_channels, out_channels):
    """
    Args:
        num_convs (int): 卷积层的数量
        in_channels (int): 输入通道的数量
        out_channels (int): 输出通道的数量
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 定义VGG网络
def vgg(conv_arch):
    """
    Args:
        conv_arch (tuple): 每个VGG块里卷积层个数和输出通道数
    """
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

# 生成网络
conv_arch = ((2, 16), (2, 32), (3, 64), (3, 128), (3, 128))
net = vgg(conv_arch)

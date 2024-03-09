'''ResNet-18 Image classfication for cifar-10 with PyTorch
Author 'Sun-qian'.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

"""每一个残差块"""


class ResidualBlock(nn.Module):  # 继承nn.Module
    def __init__(self, inchannel, outchannel, stride=1):  # __init()中必须自己定义可学习的参数
        super(ResidualBlock, self).__init__()  # 调用nn.Module的构造函数
        self.left = nn.Sequential(  # 左边，指残差块中按顺序执行的普通卷积网络
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 最常用于卷积网络中(防止梯度消失或爆炸)
            nn.ReLU(inplace=True),  # implace=True是把输出直接覆盖到输入中，节省内存
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):  # 实现前向传播过程
        out = self.left(x)  # 先执行普通卷积神经网络
        out += self.shortcut(x)  # 再加上原始x数据
        out = F.relu(out)
        return out


"""整个卷积网络，包含若干个残差块"""


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  # 设置参数为卷积的输出通道数
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 一个残差单元，每个单元中国包含2个残差块
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)  # 全连接层(1,512)-->(1,10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (
                num_blocks - 1)  # 将该单元中所有残差块的步数做成一个一个向量，第一个残差块的步数由传入参数指定，后边num_blocks-1个残差块的步数全部为1，第一个单元为[1,1]，后边三个单元为[2,1]
        layers = []
        for stride in strides:  # 对每个残差块的步数进行迭代
            layers.append(block(self.inchannel, channels, stride))  # 执行每一个残差块，定义向量存储每个残差块的输出值
            self.inchannel = channels
        return nn.Sequential(*layers)  # 如果*加在了实参上，代表的是将向量拆成一个一个的元素

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 平均池化，4*4的局部特征取平均值，最后欸(512,1,1)
        out = out.view(out.size(0), -1)  # 转换为(1,512)的格式
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

# cite - 模型定义-ResNet
# net = ResNet18()
# print(net)
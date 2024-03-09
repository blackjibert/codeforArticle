import torch
from torch import nn
import torch.nn.functional as F


# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 卷积层1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)  # 10个类别对应CIFAR-10的类别数

    def forward(self, x):
        # 第一组卷积-激活-池化层
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二组卷积-激活-池化层
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 将特征图展平
        x = x.view(-1, 64 * 8 * 8)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

# 创建模型实例
model = SimpleCNN()

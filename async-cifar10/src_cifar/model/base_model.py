import torch
from torch import nn
import torch.nn.functional as F


# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


class NetRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3*32*32 ==》6*28*28(32 + 2P - kernel + 1)
        self.pool = nn.MaxPool2d(2, 2)  # 6*28*28 ==> 6*14*14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6*14*14 ==> 16 * 10 * 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 还要一个pooling 所以输入是16 * 5 * 5 ==> 120
        self.fc2 = nn.Linear(120, 84)  # 120 ==> 84
        self.fc3 = nn.Linear(84, 10)  # 84 ==> 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


#防止过拟合
import torch.nn.functional as F

class NetRGB_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer with probability of 0.5
        self.bn1 = nn.BatchNorm2d(6)  # Add batch normalization layer after conv1
        self.bn2 = nn.BatchNorm2d(16)  # Add batch normalization layer after conv2

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after fc1
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after fc2
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

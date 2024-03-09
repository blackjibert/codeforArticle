from torch import nn
import torch.nn.functional as F
import torch


@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


class LeNetRGB_L2(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(LeNetRGB_L2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 在每一层中应用L2正则化
        self.conv1.weight_decay = weight_decay
        self.conv2.weight_decay = weight_decay
        self.fc1.weight_decay = weight_decay
        self.fc2.weight_decay = weight_decay
        self.fc3.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

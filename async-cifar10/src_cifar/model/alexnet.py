from torch import nn
import torch.nn.functional as F

#
# class AlexNettest(nn.Module):
#     def __init__(self):
#         super(AlexNettest, self).__init__()
#
#         self.cnn = nn.Sequential(
#             # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
#             # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
#             # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
#             nn.Conv2d(3, 96, 7, 2, 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, 0),
#
#             # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
#             # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
#             # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
#             nn.Conv2d(96, 256, 5, 1, 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, 0),
#
#             # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(256, 384, 3, 1, 1),
#             nn.ReLU(inplace=True),
#
#             # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(384, 384, 3, 1, 1),
#             nn.ReLU(inplace=True),
#
#             # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(384, 256, 3, 1, 1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.fc = nn.Sequential(
#             # 256个feature，每个feature 3*3
#             nn.Linear(256 * 3 * 3, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         x = self.cnn(x)
#
#         # x.size()[0]: batch size
#         # x = x.view(x.size()[0], -1)
#         # x = x.view(x.shape[0], -1)
#         x = x.view(-1, x.size(0))
#         # x = self.fc(x)
#
#         # return x
#         return F.log_softmax(x, dim=1)


# from torch import nn


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

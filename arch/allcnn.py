# Implementation taken from https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py
# This is the All-Conv-C model.
import torch.nn as nn
import torch.nn.functional as F
import torch


class AllConvNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        """
        Instantiate a All convolution neural network.

        Parameters
        ----------
        in_channels: int, default to 3.
            Number of channels in the input image. For CIFAR-10, this
            parameter is 3. Defaults to 3.
        n_classes: int, default to 10.
            Number of classes. For CIFAR-10, this parameter is
            10. Defaults to 10.
        """
        super().__init__()

        self.dp0 = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(in_channels, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.dp1 = nn.Dropout2d(p=0.5)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.dp2 = nn.Dropout2d(p=0.5)
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=0)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.bn8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, n_classes, 1)
        self.avg = nn.AvgPool2d(6)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.conv7.weight)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)

    def forward(self, x) -> torch.Tensor:
        x = self.dp0(x)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dp1(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.dp2(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)
        x = F.relu(self.conv8(x))
        x = self.bn8(x)
        x = F.relu(self.conv9(x))
        x = self.avg(x)
        x = torch.squeeze(x)
        return x

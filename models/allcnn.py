# Implementation taken from https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py
# This is the All-Conv-C model.
import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb


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
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_drop = F.dropout(x, 0.2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, 0.5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, 0.5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        out = self.softmax(pool_out)
        return out


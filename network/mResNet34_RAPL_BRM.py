"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from network import mresnet


nonlinearity = partial(F.relu, inplace=True)

class mResNet34_PHA_BRM(nn.Module):
    def __init__(self, num_classes=2):
        super(mResNet34_PHA_BRM, self).__init__()

        backbone = mresnet.resnet34(pretrained=True)

        self.firstconv = backbone.conv1_1
        self.firstbn = backbone.bn1
        self.firstrelu = backbone.relu
        self.firstmaxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.aspp = _ASPP(2048, 512, [6, 12, 18])

        self.decoder2 = nn.Sequential()
        self.decoder1 = nn.Sequential()
        self.decoder0 = nn.Sequential()

        self.bd_extract1 = nn.Sequential()
        self.bd_extract2 = nn.Sequential()
        self.bd_extract3 = nn.Sequential()

        self.segmenter = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # I will publish the whole source code after paper acception.
        # If any question, please concat with us (email: yliucit@bjtu.edu.cn).
        return segment_oup, boundary_oup


if __name__=="__main__":
    input=torch.ones(16,4,320,320)
    net=mResNet34_PHA_BRM()
    print(net)
    segment_oup, boundary_oup=net.forward(input)
    print(segment_oup.size())
    print(boundary_oup.size())

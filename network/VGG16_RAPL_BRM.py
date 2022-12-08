"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class VGG16_PHA_BRM_GF1(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_PHA_BRM_GF1, self).__init__()

        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()

        self.aspp = _ASPP(512, 512, [6,12,18])

        self.decoder2 = nn.Sequential()
        self.decoder1 = nn.Sequential()
        self.decoder0 = nn.Sequential()

        self.bd_extract1 = nn.Sequential()
        self.bd_extract2 = nn.Sequential()
        self.bd_extract3 = nn.Sequential()

        self.segmenter =  nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # I will publish the whole source code after paper acception.
        # If any question, please concat with us (email: yliucit@bjtu.edu.cn).
        return segment_oup, boundary_oup


if __name__=="__main__":
    input=torch.ones(16,4,320,320)
    net=VGG16_PHA_BRM_GF1()
    print(net)
    segment_oup, boundary_oup=net.forward(input)
    print(segment_oup.size())


import torch
import torch.nn as nn
import torchvision
from torchvision.ops import FeaturePyramidNetwork

class RCNNBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        c3, c4, c5 = 128, 256, 512

        # resnet layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   
        self.layer2 = resnet.layer2   
        self.layer3 = resnet.layer3   
        self.layer4 = resnet.layer4   

        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[c3, c4, c5],
            out_channels=out_channels
        )

        # Required by FasterRCNN
        self.out_channels = out_channels

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        c3 = self.layer2(x)      
        c4 = self.layer3(c3)     
        c5 = self.layer4(c4)     

        f = self.fpn({"c3": c3,"c4": c4, "c5": c5})
        return f


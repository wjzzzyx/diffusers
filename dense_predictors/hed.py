# from https://github.com/lllyasviel/ControlNet/blob/main/annotator/hed/__init__.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
        for i in range(1, num_layers):
            self.convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))
        self.projection = nn.Conv2d(out_channels, 1, 1)
    
    def forward(self, x, downsample=False):
        if downsample:
            x = F.max_pool2d(x, (2, 2), stride=(2, 2))
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
        return x, self.projection(x)


class HED(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

        if 'pretrained' in model_config:
            checkpoint = torch.load(model_config.pretrained, map_location='cpu')
            self.load_state_dict(checkpoint)
    
    def forward(self, x):
        h = x - self.norm
        h, proj1 = self.block1(h)
        h, proj2 = self.block2(h, downsample=True)
        h, proj3 = self.block3(h, downsample=True)
        h, proj4 = self.block4(h, downsample=True)
        h, proj5 = self.block5(h, downsample=True)

        edges = [proj1, proj2, proj3, proj4, proj5]
        edges = [F.interpolate(e, size=x.shape[2:], mode='bilinear') for e in edges]
        edges = torch.cat(edges, dim=1)
        edge = F.sigmoid(edges.mean(dim=1))
        return edge
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv


__all__ = ('DFL', 'SPPF', 'C2f', 'Bottleneck', 'MSBlockLayer', 'MSBlock', 'MSBlock_D')

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MSBlockLayer(nn.Module):
    """MSBlockLayer."""
    def __init__(self, c1, c2, k=3):
        super().__init__()
        c_ = int(c2 * 2)    # hidden channels (expand channel defualt=2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, k, g=c_)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        """Forward pass through MSBlockLayer"""
        x = self.cv1(x)
        x = self.cv2(x)
        return self.cv3(x)


class MSBlock(nn.Module):
    """MSBlock"""
    def __init__(self, c1, c2, k=3, e=1.5, n=1):
        super().__init__()
        self.c = int(c2 * e)    # e=1.5 for down sample layer
        self.g = self.c // 3    # n=3 number of MSBlockLayer
        self.cv1 = Conv(c1, self.c, 1, 1)

        # self.ms_layers = []
        # for i in range(3):
        #     if i == 0:
        #         self.ms_layers.append(nn.Identity())
        #         continue
        #     ms_layers = [MSBlockLayer(self.g, self.g, k) for _ in range(n)]
        #     self.ms_layers.append(nn.Sequential(*ms_layers))
        #     # self.ms_layers.append(nn.Sequential(*[MSBlockLayer(self.g, self.g, k) for _ in range(n)]))
        # self.ms_layers = nn.ModuleList(self.ms_layers)

        self.ms_layers = [nn.Identity()]
        self.ms_layers.extend(MSBlockLayer(self.g, self.g, k) for _ in range(2))
        self.ms_layers = nn.ModuleList(self.ms_layers)

        self.cv2 = Conv(self.c, c2, 1, 1)

    def forward(self, x):
        # y = list(self.cv1(x).split((self.g, self.g, self.g), 1))
        # ms_layers = []
        # for i, ms_layer in enumerate(self.ms_layers):
        #     x = y[i] + ms_layers[i -1] if i >= 1 else y[i]
        #     ms_layers.append(ms_layer(x))
        # return self.cv2(torch.cat(ms_layers, 1))

        x = self.cv1(x)
        layers = []
        for i, ms_layer in enumerate(self.ms_layers):
            channel = x[:, i*self.g:(i+1)*self.g,...]
            if i >=1:
                channel = channel + layers[i-1]
            channel = ms_layer(channel)
            layers.append(channel)
        return self.cv2(torch.cat(layers, 1))

        
class MSBlock_D(MSBlock):
    """MSBlock in downsample"""
    
    def __init__(self, c1, c2, k=3, e=1.5, n=1):
        super().__init__(c1, c2, k, e, n)






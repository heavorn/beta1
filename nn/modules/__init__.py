# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (DFL, SPPF, Bottleneck, C2f, MSBlockLayer, MSBlock, FasterNetLayer, GSBottleneck, VoVGSCSP, CPS_A)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, PConv, GSConv)
from .head import Detect

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'DFL', 'SPPF',
           'C2f', 'Bottleneck', 'Detect', 'Segment', 'Pose', 'Classify', 'MSBlockLayer', 'MSBlock', 'PConv', 'GSConv',
           'GSBottleneck', 'VoVGSCSP', 'FasterNetLayer', 'CPS_A')
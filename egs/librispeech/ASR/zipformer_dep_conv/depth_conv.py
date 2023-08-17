import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
from torch import Tensor, nn


class DepthConv(nn.Module):
    def __init__(self, nin: int, kernel_size: int = 3):
        super(DepthConv, self).__init__()

        self.nin = nin
        self.kernel_size = kernel_size

        self.depthwise = nn.Conv1d(nin, nin, kernel_size=3, padding=1, groups=nin)

    def forward(self, x: Tensor):
        """
        x: Tensor, (seq_len, batch, channels)
        """
        seq_len, batch, channels = x.shape
        assert channels == self.nin, f"x.shape: {x.shape}, nin: {self.nin}"

        x = x.permute(0, 2, 1)
        out = self.depthwise(x)

        assert out.shape == x.shape, f"x.shape: {x.shape}, out.shape: {out.shape}"
        return out.permute(0, 2, 1)


if __name__ == "__main__":
    dconv = DepthConv(
        nin=512,
        kernel_size=3,
    )
    inp = torch.rand((172, 50, 512))  # .permute(2, 1, 0)
    print(dconv(inp).shape)

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
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super(DepthConv, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )

    def forward(self, x: Tensor):
        """
        x: Tensor, (seq_len, batch, channels)
        """
        seq_len, batch, channels = x.shape
        assert (
            channels == self.in_channels
        ), f"x.shape: {x.shape}, in_channels: {self.in_channels}"

        x = x.permute(0, 2, 1)
        out = self.depthwise(x)

        assert out.shape == x.shape, f"x.shape: {x.shape}, out.shape: {out.shape}"
        return out.permute(0, 2, 1)


if __name__ == "__main__":
    dconv = DepthConv(
        in_channels=512,
        kernel_size=3,
    )
    inp = torch.rand((172, 50, 512))  # .permute(2, 1, 0)
    print(dconv(inp).shape)

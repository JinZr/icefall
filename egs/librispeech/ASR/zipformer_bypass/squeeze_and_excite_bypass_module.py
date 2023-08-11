import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from scaling import (
    Balancer,
    Dropout2,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    SwooshR,
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    ScheduledFloat,
    FloatLike,
    limit_param_value,
)
from torch import Tensor, nn


class PoolingModule(nn.Module):
    """
    Averages the input over the time dimension and project with a square matrix.
    """

    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.proj = ScaledLinear(
            embed_dim, bottleneck_dim, initial_scale=0.1, bias=False
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        """
        Args:
           x: a Tensor of shape (T, N, C)
          key_padding_mask: a Tensor of bool, of shape (N, T), with True in masked
            positions.
        Returns:
           a Tensor of shape (1, N, C)
        """
        if key_padding_mask is not None:
            if torch.jit.is_tracing():
                pooling_mask = (~key_padding_mask).to(x.dtype)
            else:
                pooling_mask = key_padding_mask.logical_not().to(x.dtype)  # (N, T)
            pooling_mask = pooling_mask / pooling_mask.sum(dim=1, keepdim=True)
            pooling_mask = pooling_mask.transpose(0, 1).contiguous().unsqueeze(-1)
            # now pooling_mask: (T, N, 1)
            x = (x * pooling_mask).sum(dim=0, keepdim=True).sum(dim=1, keepdim=True)
        else:
            num_frames = x.shape[0]
            pooling_mask = 1.0 / num_frames
            x = (x * pooling_mask).sum(dim=0, keepdim=True).sum(dim=1, keepdim=True)

        x = self.proj(x)
        return x


class SqueezeAndExciteBypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 16,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()

        # Replaced by batch-wise weights obtained as in def forward(self, ...)
        # self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

        self.average_pooling = PoolingModule(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
        )
        self.balancer = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=0.2,
            max_positive=0.5,
            max_abs=10.0,
        )
        self.swooshr = SwooshR()
        self.dropout = Dropout2(ScheduledFloat((0.0, 0.5), (10000.0, 0.1)))
        self.back_proj = nn.Linear(bottleneck_dim, embed_dim)

    def _get_bypass_scale(self, bypass_scale: Tensor, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return bypass_scale
        else:
            ans = limit_param_value(
                bypass_scale, min=float(self.scale_min), max=float(self.scale_max)
            )
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = (
                    torch.rand((batch_size, 1), device=ans.device)
                    < straight_through_rate
                )
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(
        self, src_orig: Tensor, src: Tensor, key_padding_mask: Optional[Tensor] = None
    ):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """

        src_pool = (
            self.average_pooling(src.permute(1, 0, 2), key_padding_mask)
            .permute(1, 0, 2)
            .squeeze(1)
            .squeeze(0)
        )
        # print("self.average_pooling(src_orig, key_padding_mask)", src_pool.shape)
        src_pool = self.balancer(src_pool)
        src_pool = self.swooshr(src_pool)
        src_pool = self.dropout(src_pool)
        src_pool = torch.sigmoid(self.back_proj(src_pool))
        # print("torch.sigmoid(self.back_proj(src_pool))", src_pool.shape)

        bypass_scale = self._get_bypass_scale(
            bypass_scale=src_pool, batch_size=src.shape[1]
        )
        return src_orig + (src - src_orig) * bypass_scale


if __name__ == "__main__":
    model = SqueezeAndExciteBypassModule(embed_dim=512, bottleneck_dim=16)

    src = torch.randn(size=(12, 110, 512))
    print(nn.Parameter(torch.full((512,), 0.5)).shape)
    mask = torch.zeros((12, 110))
    print(model(src, src, mask).shape)

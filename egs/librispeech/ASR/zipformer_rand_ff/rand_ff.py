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
    Dropout3,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    SwooshL,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
from torch import Tensor, nn

from rand_linear import *
from rand_activation_dropout_and_linear import *


class FeedforwardModuleRand(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super(FeedforwardModuleRand, self).__init__()

        # zr_modified
        # self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.in_proj = Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinearRand(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x

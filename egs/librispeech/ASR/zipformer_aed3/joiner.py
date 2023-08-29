# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn as nn
from scaling import ScaledLinear

from zipformer import CompactRelPositionalEncoding, FeedforwardModule
from cross_attention import PositionMultiheadCrossAttentionWeights, CrossAttention


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim, initial_scale=0.25)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)
        self.vocab_size = vocab_size
        self.joiner_dim = joiner_dim

        self.attn_weights = PositionMultiheadCrossAttentionWeights(
            embed_dim=joiner_dim,
            pos_dim=192,
            num_heads=5,
            query_head_dim=32,
            pos_head_dim=4,
            dropout=0.0,
        )
        self.cross_attn = CrossAttention(
            embed_dim=joiner_dim,
            num_heads=5,
            value_head_dim=12,
        )
        self.ff = FeedforwardModule(
            embed_dim=joiner_dim,
            feedforward_dim=joiner_dim * 3,
            dropout=0.1,
        )
        self.ce_out = nn.Sequential(
            nn.Linear(in_features=joiner_dim, out_features=vocab_size),
            nn.LogSoftmax(dim=-1),
        )

        # NOTE: deprecated
        self.pos_encode = CompactRelPositionalEncoding(
            embed_dim=192,
            dropout_rate=0.15,
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        lengths: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim, (
            encoder_out.shape,
            decoder_out.shape,
        )

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit

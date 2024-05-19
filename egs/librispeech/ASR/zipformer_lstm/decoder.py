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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scaling import Balancer


class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        embedding_dim: int,
        num_layers: int,
        hidden_dim: int,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        lstm_type: str = "lstm",
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          blank_id:
            The ID of the blank symbol.
          embedding_dim:
            Dimension of the input embedding.
          num_layers:
            Number of LSTM layers.
          hidden_dim:
            Hidden dimension of LSTM layers.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()

        assert lstm_type in [
            "lstm",
            "slstm",
            "mlstm",
        ], f"Unsupported lstm_type: {lstm_type}"

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,
        )
        # the balancers are to avoid any drift in the magnitude of the
        # embeddings, which would interact badly with parameter averaging.
        self.balancer = Balancer(
            embedding_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

        self.blank_id = blank_id

        self.vocab_size = vocab_size

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        if lstm_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif lstm_type == "slstm":
            from xlstm import sLSTM

            # NOTE: for sLSTM, batch_first is a default
            self.rnn = sLSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
            )
        elif lstm_type == "mlstm":
            from xlstm import mLSTM

            # NOTE: for mLSTM, batch_first is a default
            self.rnn = mLSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
            )

        self.balancer2 = Balancer(
            hidden_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, hidden_dim).
        """
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        embedding_out = self.balancer(embedding_out)

        rnn_out, (h, c) = self.rnn(embedding_out, states)
        rnn_out = F.relu(rnn_out)
        rnn_out = self.balancer2(rnn_out)
        # print(rnn_out.shape)

        return rnn_out, (h, c)

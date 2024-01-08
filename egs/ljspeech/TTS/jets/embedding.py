import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """This class implements the positional encoding
    proposed in the following paper:

    - Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf

        PE(pos, 2i) = sin(pos / (10000^(2i/d_modle))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/d_modle))

    Note::

      1 / (10000^(2i/d_model)) = exp(-log(10000^(2i/d_model)))
                               = exp(-1* 2i / d_model * log(100000))
                               = exp(2i * -(log(10000) / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """
        Args:
          d_model:
            Embedding dimension.
          dropout:
            Dropout probability to be applied to the output of this module.
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        # not doing: self.pe = None because of errors thrown by torchscript
        self.pe = torch.zeros(1, 0, self.d_model, dtype=torch.float32)

    def extend_pe(self, x: torch.Tensor) -> None:
        """Extend the time t in the positional encoding if required.

        The shape of `self.pe` is (1, T1, d_model). The shape of the input x
        is (N, T, d_model). If T > T1, then we change the shape of self.pe
        to (N, T, d_model). Otherwise, nothing is done.

        Args:
          x:
            It is a tensor of shape (N, T, C).
        Returns:
          Return None.
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model, dtype=torch.float32)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Now pe is of shape (1, T, d_model), where T is x.size(1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
          x:
            Its shape is (N, T, C)

        Returns:
          Return a tensor of shape (N, T, C)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScaledPositionalEncoding(nn.Module):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def extend_pe(self, x: torch.Tensor) -> None:
        """Extend the time t in the positional encoding if required.

        The shape of `self.pe` is (1, T1, d_model). The shape of the input x
        is (N, T, d_model). If T > T1, then we change the shape of self.pe
        to (N, T, d_model). Otherwise, nothing is done.

        Args:
          x:
            It is a tensor of shape (N, T, C).
        Returns:
          Return None.
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model, dtype=torch.float32)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Now pe is of shape (1, T, d_model), where T is x.size(1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)

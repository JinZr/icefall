import torch
from embedding import PositionalEncoding
from encoder_layer import EncoderLayer, LayerNorm
from multi_layer_conv import MultiLayeredConv1d
from repeat import repeat
from torch import nn


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in selfattention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=nn.Module,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        stochastic_depth_rate=0.0,
    ):
        """Construct an Encoder object."""
        super(TransformerEncoder, self).__init__()

        self.conv_subsampling_factor = 1
        self.embed = torch.nn.Sequential(
            input_layer,
            pos_enc_class(attention_dim, positional_dropout_rate),
        )
        self.normalize_before = normalize_before

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                self_attn=nn.MultiheadAttention,
                attention_heads=attention_heads,
                attention_dropout_rate=attention_dropout_rate,
                feed_forward=MultiLayeredConv1d,
                feed_forward_dim=attention_dim,
                feed_forward_linear_units=linear_units,
                feed_forward_dropout_rate=dropout_rate,
                feed_forward_positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
                stochastic_depth_rate=stochastic_depth_rate
                * float(1 + lnum)
                / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        xs = self.embed(xs)

        for _, encoder_layer in enumerate(self.encoders):
            xs, masks = encoder_layer(xs, masks)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache

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

from zipformer import CompactRelPositionalEncoding


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class PositionMultiheadCrossAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.

        self.query_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25
        )
        self.key_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        # NOTE: [deprecated] linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )
        # NOTE: we use abs-pos-enc instead
        self.decoder_pos = PositionalEncoding(d_model=embed_dim)

        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            query & key: input of shape (seq_len, batch_size, embed_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        key = self.key_proj(key)

        # NOTE: apply abs pos emb to query
        query = self.decoder_pos(query)
        query = self.query_proj(query)

        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        # seq_len, batch_size, _ = x.shape
        (
            query_len,
            batch_size,
            _,
        ) = query.shape  # actual dim: (query_len, batch, _)
        (
            key_len,
            _,
            _,
        ) = key.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = query[..., 0:query_dim]
        k = key[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = key[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(query_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(key_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(key_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # We can't put random.random() in the same line
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True
        use_pos_scores = False

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * query_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            pos_scores = pos_scores.as_strided(
                (num_heads, batch_size, query_len, key_len),
                (
                    pos_scores.stride(0),
                    pos_scores.stride(1),
                    pos_scores.stride(2) - pos_scores.stride(3),
                    pos_scores.stride(3),
                ),
                storage_offset=pos_scores.stride(3) * (query_len - 1),
            )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=25.0, penalty=1.0e-04, name=self.name
            )

        assert attn_scores.shape == (num_heads, batch_size, query_len, key_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                key_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        left_context_len: int,
        key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, left_context_len+2*seq_len-1, pos_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            left_context_len: number of left context frames.
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
              are True in this mask will be ignored as sources in the attention weighting.

        Returns:
           - attention weights, of shape (hum_heads, batch_size, seq_len, seq_len2),
             interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
           - updated cached attention key tensor of left context.
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim

        # Pad cached left contexts
        assert cached_key.shape[0] == left_context_len, (
            cached_key.shape[0],
            left_context_len,
        )
        k = torch.cat([cached_key, k], dim=0)
        # Update cached left contexts
        cached_key = k[-left_context_len:, ...]

        # The length of key
        k_len = k.shape[0]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(k_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
            2, 0, 3, 1
        )
        # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

        # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
        #  [where seq_len2 represents relative position.]
        pos_scores = torch.matmul(p, pos_emb)

        if torch.jit.is_tracing():
            (num_heads, batch_size, time1, n) = pos_scores.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(k_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_scores = pos_scores.reshape(-1, n)
            pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
            pos_scores = pos_scores.reshape(num_heads, batch_size, time1, k_len)
        # the following .as_strided() expression converts the last axis of pos_scores from relative
        # to absolute position.  I don't know whether I might have got the time-offsets backwards or
        # not, but let this code define which way round it is supposed to be.
        else:
            pos_scores = pos_scores.as_strided(
                (num_heads, batch_size, seq_len, k_len),
                (
                    pos_scores.stride(0),
                    pos_scores.stride(1),
                    pos_scores.stride(2) - pos_scores.stride(3),
                    pos_scores.stride(3),
                ),
                storage_offset=pos_scores.stride(3) * (seq_len - 1),
            )

        attn_scores = attn_scores + pos_scores

        assert attn_scores.shape == (
            num_heads,
            batch_size,
            seq_len,
            k_len,
        ), attn_scores.shape

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, k_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        attn_weights = attn_scores.softmax(dim=-1)

        return attn_weights, cached_key

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .mean(dim=(1, 2))
                )
                logging.info(
                    f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}"
                )


class CrossAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (am_seq_len, batch_size, embed_dim) = x.shape
        (_, _, lm_seq_len, _) = attn_weights.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (
            num_heads,
            batch_size,
            lm_seq_len,
            am_seq_len,
        ), f"{attn_weights.shape} {x.shape}"

        x = self.in_proj(x)  # (am_seq_len, batch_size, num_heads * value_head_dim)
        # print("projected x.shape", x.shape)

        x = x.reshape(am_seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, am_seq_len, value_head_dim)
        # print("permuted x.shape", x.shape)

        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, lm_seq_len, value_head_dim)
        # print("attended x.shape", x.shape)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(lm_seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (lm_seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)
        # print("returned x.shape", x.shape)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor, of shape (seq_len, batch_size, embed_dim)
            attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
              with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
              attn_weights.sum(dim=-1) == 1.
            cached_val: cached attention value tensor of left context,
              of shape (left_context_len, batch_size, value_dim)
            left_context_len: number of left context frames.

        Returns:
           - attention weighted output, a tensor with the same shape as x.
           - updated cached attention value tensor of left context.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len2)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Pad cached left contexts
        assert cached_val.shape[0] == left_context_len, (
            cached_val.shape[0],
            left_context_len,
        )
        x = torch.cat([cached_val, x], dim=0)
        # Update cached left contexts
        cached_val = x[-left_context_len:, ...]

        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x, cached_val


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
            Its shape is (T, N, C)

        Returns:
          Return a tensor of shape (T, N, C)
        """
        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        self.extend_pe(x)
        x = x + self.dropout(self.pe[:, : x.size(1), :])
        return x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)


if __name__ == "__main__":
    attn = PositionMultiheadCrossAttentionWeights(
        embed_dim=512,
        pos_dim=192,
        num_heads=5,
        query_head_dim=32,
        pos_head_dim=4,
        dropout=0.0,
    )
    pos_encode = CompactRelPositionalEncoding(
        embed_dim=192,
        dropout_rate=0.15,
    )
    self_attn = CrossAttention(
        embed_dim=512,
        num_heads=5,
        value_head_dim=12,
    )

    # print("__main__ === for inference ===")
    # am : [T, B, encoder_dim]
    # lm : [1, B, decoder_dim]
    # am = torch.rand(100, 2, 512)
    # lm = torch.rand(10, 2, 512)
    # q / K separate seq_len

    # weights = RelPositionMultiheadAttentionWeights()
    # attn = CrossAttention(512, 5, 12)
    # attn_weights = weights(lm, am, pos_emb)
    # print("weights(am_pruned, lm_pruned, pos_emb).shape", attn_weights.shape)
    # res = attn(am, attn_weights)
    # res = attn(am, lm, torch.Tensor([70, 80]), None)
    # print("__main__ res", res.shape)

    print("__main__ === for training ===")
    # am : [B, T,  encoder_dim]
    # lm : [B, T,  decoder_dim]
    am_pruned = torch.rand(100, 2, 512)
    lm_pruned = torch.rand(50, 2, 512)
    pos_emb = pos_encode(lm_pruned)

    res = attn(query=lm_pruned, key=am_pruned, pos_emb=pos_emb)
    a = self_attn(am_pruned, res)
    print("__main__ res", res.shape)
    print("__main__ a", a.shape)

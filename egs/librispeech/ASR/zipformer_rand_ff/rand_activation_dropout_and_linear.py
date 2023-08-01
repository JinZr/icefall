import k2

import torch
from torch import Tensor, nn
from torch.nn import init, Parameter
import torch.nn.functional as F

from scaling import limit_param_value, ScaledLinear

from rand_linear import Linear


class ActivationDropoutAndLinearFunctionRand(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        activation: str,
        dropout_p: float,
        dropout_shared_dim: Optional[int],
    ):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            # else it won't be very memory efficient.
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None

        ctx.save_for_backward(x, weight, bias, dropout_mask)

        ctx.activation = activation

        forward_activation_dict = {
            "SwooshL": k2.swoosh_l_forward,
            "SwooshR": k2.swoosh_r_forward,
        }
        # it will raise a KeyError if this fails.  This will be an error.  We let it
        # propagate to the user.
        activation_func = forward_activation_dict[activation]
        x = activation_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        forward_and_deriv_activation_dict = {
            "SwooshL": k2.swoosh_l_forward_and_deriv,
            "SwooshR": k2.swoosh_r_forward_and_deriv,
        }
        # the following lines a KeyError if the activation is unrecognized.
        # This will be an error.  We let it propagate to the user.
        func = forward_and_deriv_activation_dict[ctx.activation]

        y, func_deriv = func(x)
        if dropout_mask is not None:
            y = y * dropout_mask
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        if dropout_mask is not None:
            # order versus func_deriv does not matter
            x_deriv = x_deriv * dropout_mask

        return x_deriv, weight_deriv, bias_deriv, None, None, None


class ActivationDropoutAndLinearRand(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        l = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )

        self.weight = l.weight
        self.weight.requires_grad = False
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", l.bias)
        self.scalar = torch.nn.Parameter(torch.ones(1))
        self.scalar.requires_grad = True

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
            else:
                assert False, self.activation
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunctionRand.apply(
            x,
            self.weight
            * limit_param_value(self.scalar, 0.1, 10, training=self.training),
            self.bias,
            self.activation,
            float(self.dropout_p),
            self.dropout_shared_dim,
        )

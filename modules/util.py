import os
import platform

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from utils.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from utils.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

# class EqualLinear(nn.Module):
#     def __init__(
#         self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
#     ):
#         super().__init__()

#         self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

#         else:
#             self.bias = None

#         self.activation = activation
#         self.device = device

#         self.scale = (1 / math.sqrt(in_dim)) * lr_mul
#         self.lr_mul = lr_mul

#     def forward(self, input):
#         if self.activation:
#             out = F.linear(input, self.weight * self.scale)
#             out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)

#         else:
#             out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

#         return out

#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
#         )

# class FusedLeakyReLUFunctionBackward(Function):
#     @staticmethod
#     def forward(ctx, grad_output, out, negative_slope, scale):
#         ctx.save_for_backward(out)
#         ctx.negative_slope = negative_slope
#         ctx.scale = scale

#         empty = grad_output.new_empty(0)

#         grad_input = fused.fused_bias_act(
#             grad_output, empty, out, 3, 1, negative_slope, scale
#         )

#         dim = [0]

#         if grad_input.ndim > 2:
#             dim += list(range(2, grad_input.ndim))

#         grad_bias = grad_input.sum(dim).detach()

#         return grad_input, grad_bias

#     @staticmethod
#     def backward(ctx, gradgrad_input, gradgrad_bias):
#         out, = ctx.saved_tensors
#         gradgrad_out = fused.fused_bias_act(
#             gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
#         )

#         return gradgrad_out, None, None, None



# class FusedLeakyReLUFunction(Function):
#     @staticmethod
#     def forward(ctx, input, bias, negative_slope, scale):
#         empty = input.new_empty(0)
#         out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
#         ctx.save_for_backward(out)
#         ctx.negative_slope = negative_slope
#         ctx.scale = scale

#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         out, = ctx.saved_tensors

#         grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
#             grad_output, out, ctx.negative_slope, ctx.scale
#         )

#         return grad_input, grad_bias, None, None

# def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, device='cpu'):
#     if platform.system() == 'Linux' and torch.cuda.is_available() and device != 'cpu':
#         return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
#     else:
#         return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)
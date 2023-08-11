from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.modules.rnn import LSTM

class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, padding: Union[int, tuple] = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y


class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#
#         self.linear1 = nn.Linear(dim, hidden_dim)
#         self.lstm = LSTM(dim, hidden_dim, 1, bidirectional=True)
#         self.activate = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(hidden_dim*2, dim)
#
#
#     def forward(self, x):
#         # x = self.linear1(x)
#         x, _ = self.lstm(x)
#         x = self.activate(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         x = self.dropout(x)
#         return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerVit(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims).contiguous()
        # return torch.permute(x, self.dims)


class Unsqueeze(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dim (int): The desired ordering of dimensions
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.unsqueeze(x, self.dim)


class Squeeze(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dim (int): The desired ordering of dimensions
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        assert self.dim < x.ndim
        return torch.squeeze(x, self.dim)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Attention(nn.Module):
    def __init__(self, attn_drop=0.):
        super(Attention, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, scale):
        # todo: attn = (q @ k.transpose(0, 1)) * scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        q = attn @ v

        q = einops.rearrange(q, 'c h fph -> c (h fph)')
        return q


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, use_bias=False, attn_drop=0., proj_drop=0., attn_type=1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_type = attn_type
        if attn_type == 1:
            self.w_q = nn.Linear(dim, dim, bias=use_bias)
            self.w_k = nn.Linear(dim, dim, bias=use_bias)
            self.w_v = nn.Linear(dim, dim, bias=use_bias)
        else:
            self.qkv = nn.Linear(dim, 3 * dim, bias=use_bias)

        self.attention = Attention(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        C, G = x.shape
        if self.attn_type == 1:
            q = self.w_q(x).reshape(C, self.num_heads, G // self.num_heads)
            k = self.w_k(x).reshape(C, self.num_heads, G // self.num_heads)
            v = self.w_v(x).reshape(C, self.num_heads, G // self.num_heads)
        else:
            qkv = self.qkv(x).reshape(C, 3, self.num_heads, G // self.num_heads)
            q, k, v = qkv.unbind(1)

        x = self.attention(q, k, v, self.scale)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

temp1 = torch.rand((10, 100))
model = MultiHeadAttention(100)
print(model(temp1).shape)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, use_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(dim / num_heads)

        self.w_q = nn.Linear(dim, dim, bias=use_bias)
        self.w_k = nn.Linear(dim, dim, bias=use_bias)
        self.w_v = nn.Linear(dim, dim, bias=use_bias)

        self.attention = Attention(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, F = q.shape
        q = self.w_q(q).reshape(B, self.num_heads, F // self.num_heads)
        k = self.w_k(k).reshape(B, self.num_heads, F // self.num_heads)
        v = self.w_v(v).reshape(B, self.num_heads, F // self.num_heads)

        q = self.attention(q, k, v, self.scale)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q
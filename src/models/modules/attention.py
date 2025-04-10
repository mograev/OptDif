"""
Attention modules used in Stable Diffusion.
Source: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/attention.py
"""

from inspect import isfunction
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from src.models.modules.utils import checkpoint


def exists(val):
    """Check if a value exists (is not None)"""
    return val is not None


def uniq(arr):
    """Get unique elements from an array"""
    return{el: True for el in arr}.keys()


def default(val, d):
    """Return default value if val is None, otherwise return val"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    """Get maximum negative value for a tensor"""
    return -torch.finfo(t.dtype).max


def init_(tensor):
    """Initialize a tensor with uniform distribution"""
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    """
    Feedforward layer with Gated Linear Unit (GLU) activation.
    """
    def __init__(self, dim_in, dim_out):
        """
        Initialize GEGLU layer.
        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """
        Forward pass through the GEGLU layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying GEGLU activation.
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Feedforward layer with optional GLU activation.
    """
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        """
        Initialize FeedForward layer.
        Args:
            dim (int): Input dimension.
            dim_out (int): Output dimension. If None, defaults to dim.
            mult (int): Multiplier for inner dimension.
            glu (bool): Whether to use GLU activation.
            dropout (float): Dropout rate.
        """
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        """Forward pass through the FeedForward layer."""
        return self.net(x)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    """Normalize layer for the input channels."""
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    """
    Linear attention layer.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Initialize LinearAttention layer.
        Args:
            dim (int): Input dimension.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
        """
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Forward pass through the LinearAttention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor after applying linear attention.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention layer.
    """
    def __init__(self, in_channels):
        """
        Initialize SpatialSelfAttention layer.
        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        """
        Forward pass through the SpatialSelfAttention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    """
    Cross-attention layer.
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        """
        Initialize CrossAttention layer.
        Args:
            query_dim (int): Dimension of the query.
            context_dim (int): Dimension of the context. If None, defaults to query_dim.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            dropout (float): Dropout rate.
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Forward pass through the CrossAttention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            context (torch.Tensor): Context tensor of shape (B, N, C). If None, defaults to x.
            mask (torch.Tensor): Optional attention mask.
        Returns:
            torch.Tensor: Output tensor after applying cross-attention.
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    Basic transformer block with two cross-attention layers and a feedforward layer.
    """
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        """
        Initialize BasicTransformerBlock.
        Args:
            dim (int): Input dimension.
            n_heads (int): Number of attention heads.
            d_head (int): Dimension of each attention head.
            dropout (float): Dropout rate.
            context_dim (int): Dimension of the context. If None, defaults to dim.
            gated_ff (bool): Whether to use gated feedforward layer.
            checkpoint (bool): Whether to use checkpointing for memory efficiency.
        """
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        """
        Forward pass through the BasicTransformerBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            context (torch.Tensor): Context tensor of shape (B, N, C). If None, defaults to x.
        Returns:
            torch.Tensor: Output tensor after applying transformer block.
        """
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        """
        Forward pass through the BasicTransformerBlock without checkpointing.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            context (torch.Tensor): Context tensor of shape (B, N, C). If None, defaults to x.
        Returns:
            torch.Tensor: Output tensor after applying transformer block.
        """
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding)
    and reshape to b, t, d. Then apply standard transformer action. Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        """
        Initialize SpatialTransformer.
        Args:
            in_channels (int): Number of input channels.
            n_heads (int): Number of attention heads.
            d_head (int): Dimension of each attention head.
            depth (int): Number of transformer blocks.
            dropout (float): Dropout rate.
            context_dim (int): Dimension of the context. If None, defaults to in_channels.
        """
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        """
        Forward pass through the SpatialTransformer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            context (torch.Tensor): Context tensor of shape (B, N, C). If None, defaults to x.
        Returns:
            torch.Tensor: Output tensor after applying transformer block.
        """
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
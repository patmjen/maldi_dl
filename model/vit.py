from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from model.init import init_weights_truncnormal, init_weights_xavier


def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViTNoPad(nn.Module):
    def __init__(
        self, *,
        seq_len,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.0,
        init_norm = 'patch',
    ):
        super().__init__()

        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        if init_norm == 'spectrum':
            spectrum_norm = nn.LayerNorm(seq_len)
            patch_norm = nn.Identity()
        elif init_norm == 'patch':
            spectrum_norm = nn.Identity()
            patch_norm = nn.LayerNorm(patch_dim)
        elif init_norm == 'none':
            spectrum_norm = nn.Identity()
            patch_norm = nn.Identity()
        else:
            raise ValueError(f'Unknown init_norm: {init_norm}')

        self.to_patch_embedding = nn.Sequential(
            spectrum_norm,
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            patch_norm,
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)


class SimpleViT(nn.Sequential):
    def __init__(self, hparams):
        super().__init__()
        assert hparams.model.classname == 'vit'
        patch_size = hparams.model.patch_size
        seq_len = hparams.data.spectrum_len
        if 'should_pad' in hparams.model:
            should_pad = hparams.model.should_pad
        else:
            should_pad = True
        pad = nn.Identity()
        if seq_len % patch_size != 0:
            pad_len = patch_size - (seq_len % patch_size)
            seq_len += pad_len
            if should_pad:
                pad = nn.ConstantPad1d((0, pad_len), 0)

        vit = SimpleViTNoPad(
            seq_len=seq_len,
            patch_size=hparams.model.patch_size,
            num_classes=len(hparams.data.classes),
            dim=hparams.model.dim,
            depth=hparams.model.depth,
            heads=hparams.model.heads,
            mlp_dim=hparams.model.mlp_dim,
            channels=hparams.model.channels,
            dropout=hparams.model.dropout,
            init_norm=hparams.model.init_norm,
        )

        if hparams.model.weight_init == 'xavier':
            vit.apply(init_weights_xavier)
        elif hparams.model.weight_init == 'trunc_normal':
            vit.apply(init_weights_truncnormal)

        self.append(pad)
        self.append(vit)

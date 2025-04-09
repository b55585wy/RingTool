import torch
from torch import nn
from einops import rearrange, repeat
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Seq_Transformer(nn.Module):
    def __init__(self, n_channel, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        x = self.position(x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t


class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        super(Transformer, self).__init__()
        self.backbone = backbone
        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if backbone == False:
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.transformer(x)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
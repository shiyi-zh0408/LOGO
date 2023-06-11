import torch
import math
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, linear_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # QKV matrix
        self.q_matrix = nn.Linear(linear_dim, linear_dim, bias=qkv_bias)
        self.k_matrix = nn.Linear(linear_dim, linear_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q_in, k_in, x):
        B, N, C = x.shape
        q = self.q_matrix(q_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        k = self.k_matrix(k_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,num_heads,N,N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,N,C
        x = x + (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,N,C
        x = self.bn(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        q = q.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        k = k.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        return q, k, x, attn


class Encoder_Blocks(nn.Module):
    def __init__(self, qk_dim, dim, linear_dim, num_heads, num_layers, attn_drop=0., proj_drop=0.):
        super(Encoder_Blocks, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(Attention(dim, linear_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop))
        self.model = nn.ModuleList(model_list)
        self.linear_q = nn.Linear(qk_dim, linear_dim)
        self.linear_k = nn.Linear(qk_dim, linear_dim)
        self.qk_dim = qk_dim

    def forward(self, q, k, x):
        attn_qk = 0
        q = self.linear_q(q)
        k = self.linear_k(k)
        for i, _layer in enumerate(self.model):
            q, k, x, attn = _layer(q, k, x)
            if i == 3:
                attn_qk = attn
        return x, attn_qk


def temporal_position_encoding(size):
    bs = size[0]
    max_len = size[1]
    d_model = size[2]
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    pe_b = torch.cat([pe for i in range(bs)])
    return pe_b

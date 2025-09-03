from attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

class MHA(nn.Module):
    def __init__(self, D, nh):
        self.c_attn = nn.Linear(D, 3 * D)
        self.head_dim = D // nh
        self.nh = nh
    
    def forward(self, x):
        B, T, D = x.size()
        combined_QKV = self.c_attn(x) # (B, T, D) -> [QQQ, KKK, VVV]
        q, k, v = combined_QKV.split(D, dim=2)
        q = q.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, T, nh, head_dim) -> (B, nh, T, head_dim)
        k = k.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, T, nh, head_dim) -> (B, nh, T, head_dim)
        v = v.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, T, nh, head_dim) -> (B, nh, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, D)
        return out

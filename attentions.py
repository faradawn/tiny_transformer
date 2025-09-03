import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, D, nh):
        super().__init__()
        self.c_attn = nn.Linear(D, 3 * D)
        self.head_dim = D // nh # head dimension is model size divided by num of head
        self.nh = nh
    def forward(self, x):
        B, T, D = x.size()
        
        combined_QKV = self.c_attn(x) # (B, T, 3D) -> [QQ, KK, VV]
        q, k, v = combined_QKV.split(D, dim=2) # split on last dim -> each is (B, T, D) and D = nh * head_dim
        q = q.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, T, nh, head_dim) -> (B, nh, T, head_dim)
        k = k.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, nh, T, head_dim)
        v = v.view(B, T, self.nh, self.head_dim).transpose(1,2) # (B, nh, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # attention shape (B, nh, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=bool), diagonal=1)
        mask = mask.view(1, 1, T, T)
        att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        out = att @ v # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out
        

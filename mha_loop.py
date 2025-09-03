from attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

class MHA_loop(nn.Module):
    def __init__(self, D, nh): # nh is number of heads
        self.head_dim = D // nh
        self.nh = nh
        self.heads = nn.ModuleList([Attention(D) for _ in range(nh)])
        self.out_proj = nn.Linear(nh * self.head_dim, D)

    def forward(self, x):
        head_ouputs = []
        for head in self.heads:
            single_output = head(x)
            head_ouputs.append(single_output)

        out = torch.cat(head_ouputs, dim=-1)
        out = self.out_proj(out)
        return out
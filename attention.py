from attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

class Attention(nn.Module):
    def __init__(self, D): # model size 
        super().__init__()
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        att = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))
        att = F.softmax(att, dim=-1)
        out = att @ V
        return out

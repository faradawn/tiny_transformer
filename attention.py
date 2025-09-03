import torch
import math
import torch.nn.functional as F
T = 5
D = 16
Q = torch.randint(0, 3, (T, D), dtype=torch.long)
K = torch.randint(0, 3, (3, D), dtype=torch.long)

d_k = K.size(-1)
print(d_k)
att = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
print("QK",att) # [T, 3]
att = F.softmax(att, dim=-1)
print("after softmax", att) # [T, 1]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

"""
Transfomer

logits

--- transformer block x n
MLP
layer norm
attention
layer norm
---

pos_emb
token_emb

x
"""

class Attention(nn.Module):
    def __init__(self, D, d_head) -> None: # D is model dimension, d_head is hidden dimension (intermediate size)
        super().__init__()
        self.W_q = nn.Linear(D, d_head)
        self.W_k = nn.Linear(D, d_head)
        self.W_v = nn.Linear(D, d_head)
        self.att_dropout = nn.Dropout()
        self.last_dropout = nn.Dropout()

    def forward(self, x):
        B, T, D = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        att = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1)) # swaps the last two dimensions
        mask = torch.triu(torch.ones(T, T, dtype=bool), diagonal=1)
        mask = mask.view(1, T, T)
        att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        out = att @ V
        out = self.last_dropout(out)
        return out

class MLP(nn.Module):
    def __init__(self, d_inter, d_ff = None): 
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_inter

        self.W_in = nn.Linear(d_inter, d_ff)
        self.gelu = nn.GELU()
        self.W_out = nn.Linear(d_ff, d_inter)
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = self.W_in(x)
        x = self.gelu(x)
        x = self.W_out(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, d_inter, d_ff) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_inter)
        self.att = Attention(d_inter, d_head)
        self.layer_norm_2 = nn.LayerNorm(d_inter)
        self.mlp = MLP(d_inter, d_ff)
    
    def forward(self, x):
        x = x + self.att(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, n_layers, d_model, d_head, d_inter, d_ff, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_inter)
        self.pos_emb = nn.Embedding(vocab_size, d_inter)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, d_head, d_inter, d_ff) for _ in range(n_layers)])
        self.logits_layer = nn.Linear(d_inter, vocab_size)
    
    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for layer in self.transformer_blocks:
            x = layer(x)
        
        logits = self.logits_layer(x)
        return logits

if __name__ == "__main__":
    n_layers = 2 
    d_model = 16 # hidden size
    d_head = d_model # q k v_head_dim
    d_inter = d_model # intermediate_size
    d_ff = 128 # moe_intermediate_size
    vocab_size = 129280 # vocab_size
    epoches = 10
    gradient_accumulation_steps = 4

    # hyper-parameters for batching
    batch_size = 32         # number of sequences processed in parallel
    block_size = 128        # length of each sequence ("time" dimension)

    DATA_DIR = os.path.join(os.path.dirname(__file__), "shakespear")
    train_data_np = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
    val_data_np   = np.fromfile(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16)

    train_data = torch.from_numpy(train_data_np.astype(np.int64))
    val_data   = torch.from_numpy(val_data_np.astype(np.int64))

    print("train shape", train_data.size())
    print("val shape", val_data.size())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_batch(split: str):
        # Get a random batch
        data = train_data if split == 'train' else val_data

        # random starting indices for each sequence in the batch
        ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))

        # stack the slices into (B, T) tensors
        x = torch.stack([data[i:i + block_size]       for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])

        return x.to(device), y.to(device)
    
    # Define model
    model = TransformerModel(n_layers, d_model, d_head, d_inter, d_ff, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for e in range(epoches):
        x, y = get_batch('train')    
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()
        print("loss", loss)
        optimizer.step()
        optimizer.zero_grad()
    # save model
    print("done training")

    





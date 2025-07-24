from torch import nn

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
    def __init__(self, D, d_head) -> None: # D is model dimension, F is hidden dimension (intermediate size)
        super().__init__()
        self.W_q = nn.Linear(D, d_head)
        self.W_k = nn.Linear(D, d_head)
        self.W_v = nn.Linear(D, d_head)

    def forward(self, x):
        x = layernorm(x)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        att = softmax(Q @ K.T / sqrt(F))
        att = dropout(att)
        out = att @ V
        return out

class MLP(nn.Moudle):
    def __init__(self, d_inter, d_ff): # Low rank compression
        self.W_in = nn.Linear(d_inter, d_ff)
        self.W_out = nn.Linear(d_ff, d_inter)
    
    def forward(self, x):
        x = layernorm(x)
        mask = lower_triangle_mask() # TODO: is this the right place to place the mask? 
        x = self.W_out(self.W_in(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, d_inter, d_ff) -> None:
        super().__init__()
        self.att = Attention(d_model, d_head)
        self.mlp = MLP(d_inter, d_ff)
    
    def forward(self, x):
        x = self.att(x)
        x = self.mlp(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, n_layers, d_model, d_head, d_inter, d_ff, vocab_size):
        self.token_emb = nn.Embedding(d_model)
        self.pos_emb = nn.Embedding(d_model)
        self.transformer_blocks = [TransformerBlock(d_model, d_head, d_inter, d_ff) for _ in range(n_layers)]
        self.logits_layer = nn.Linear(d_inter, vocab_size)
    
    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(x)
        for layer in self.transformer_blocks:
            x = layer(x)
        
        logits = self.logits_layer(x)
        return logits

if __name__ == "__main__":
    n_layers = 61 
    d_model = 7168 # hidden size
    d_head = 128 # q k v_head_dim
    d_inter = 18432 # intermediate_size
    d_ff = 2048 # moe_intermediate_size
    vocab_size = 129280 # vocab_size
    epoches = 10

    model = TransformerModel(n_layers, d_model, d_head, d_inter, d_ff, vocab_size)

    optimizer = Adam(lr=1e-3)

    # Data (x, y), x is token ids of size (B, T). y is one_hot encoding of logit (0,0,0, 1, 0,0,0) of the correct next token
    batches = []


    # Training loop
    for e in range(epoches):
        for mini_batch in batches:
            optimizer.no_grad()
            x, y = mini_batch
            logits = model(x)
            loss = crossEntropy(y, logits)
            loss.backward()
            optimizer.step()

    # save model
    print("done trainign")





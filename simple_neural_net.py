import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, layers, d_in, d_hidden, d_out):
        super().__init__()
        self.layers = nn.ModuleList()
        d_prev = d_in
        for _ in range(layers):
            layer = nn.Linear(d_prev, d_hidden) # Wx + b
            d_prev = d_hidden
            self.layers.append(layer)
        self.out = nn.Linear(d_prev, d_out)
            
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.relu(x)
        return self.out(x)


if __name__ == "__main__":
    # training loop
    B = 16
    T = 128 # sequence length
    D = 256 # model size / hidden dimension size
    d_model = 512
    VOCAB_SIZE = 10500
    num_epoch = 10
    
    model = Model(layers=2, d_in=D, d_hidden=d_model, d_out=VOCAB_SIZE)

    num_batches = 2
    batches = []
    for _ in range(num_batches):
        x = torch.randn(B * T, D)
        y = torch.randn((B * T, VOCAB_SIZE))
        batches.append((x, y))


    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epoch):
        print(f"epoch {epoch}")
        for batch in batches: # mini-batch
            opt.zero_grad()
            x, y = batch
            y_pred = model.forward(x)
            loss = torch.mean(torch.square(y - y_pred))
            loss.backward()
            opt.step()
            print(f"    loss {loss}")




        
    
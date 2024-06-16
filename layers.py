import torch
from torch import nn
import torch.nn.functional as F
import math

    
class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embeds = self.embedding(x)
        return embeds
    

class PositionalEmbedding(nn.Module):
    '''
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    '''
    def __init__(self,
                 d_model,
                 max_positions=5000):
        super().__init__()

        self.pos_encodings = torch.zeros(max_positions, d_model)
        positions = torch.arange(max_positions).unsqueeze(-1)

        # Use log for numerical stability
        denom = torch.exp(math.log(10000) * (torch.arange(0, d_model, 2) / d_model)).unsqueeze(0) 

        self.pos_encodings[:, ::2] = torch.sin(positions/denom) # multiplication better?
        self.pos_encodings[:, 1::2] = torch.cos(positions/denom)

        self.pos_encodings.requires_grad = False

    def forward(self, x):
        x = x + self.pos_encodings[:x.size()[1], :] # requires grad false? 
        return x
    

class PositionwiseFFN(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim) -> None:
        super().__init__()
        self.feedforward_in = nn.Linear(d_model, h_dim)
        self.feedforward_out = nn.Linear(h_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.feedforward_in(x))
        z = self.feedforward_out(h)
        return z
    

class LayerNorm(nn.Module):
    def __init__(self,
                 d_model,
                 eps=1e-5): # What is the default in the transformer paper?
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x is (B, S, d_model)
        mean = torch.mean(x, -1).unsqueeze(-1) # (B, S, 1)
        std = torch.std(x, -1).unsqueeze(-1) # (B, S, 1)
        x_norm = (x - mean)/(std + self.eps) * self.gamma + self.beta
        return x_norm
    

class LM_head(nn.Module):
    def __init__(self,
                 d_model,
                 vocab_size):
        super().__init__()
        
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        logits = self.linear(x)
        out = F.softmax(logits, -1) 
        return out       
import torch
from torch import nn
import torch.functional as F

class Transformer(nn.Module):
    pass

# class Encoder(nn.Module):
#     def __init__(self, encoder, decoder):
    
#         self.encoder = encoder
        
#     def forward(self, x):
#         z = self.encoder(x)
#         y = self.decoder(z, y_tm1)

# class Decoder(nn.Module):
#     pass

class Encoder(nn.Module):
    def __init__(self,
                 N_encoder,
                 N_decoder):
        super().__init__()


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        


class EncoderBlock(nn.Module):
    def __init__(self,
                 feedforward,
                 attention,
                 norm):
        super().__init__()

        self.feedforward = feedforward
        self.attention = attention
        self.norm = norm 
    
    def forward(self, x):

        # Apply self-attention
        x = self.norm(x + self.attention(x, x, x))

        # Apply FF layer
        x = self.norm(x + self.feedforward(x))

        return x

class DecoderBlock(nn.Module):
    def __init__(self,
                 feedforward,
                 attention,
                 norm,
                 mask):
        super().__init__()

        self.feedforward = feedforward
        self.attention = attention
        self.norm = norm
        self.mask = mask 

    def forward(self, x, z):

        # Apply self-attention
        x = self.norm(x + self.attention(x, x, x, self.mask))

        # Apply cross-attention
        x = self.norm(x + self.attention(x, z, z, self.mask))

        # Apply FF layer
        x = self.norm(x + self.feedforward(x))

        return x
    
        
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

class attention(nn.Module):
    def __init__(self,
                 d_model):
        super().__init__()

        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model) # input of size (B, S, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
    
    def compute_attention(self, q, k, v, mask):
        # Implment softmax(QK^T/sqrt(d_k))V
        d_k = k.shape[-1]

        M = torch.matmul(q, k.mT)/torch.sqrt(d_k) # (B, S, d_model) x (B, d_model, S) = (B, S, S)

        # Mask should only be used for decoder 
        if mask:
            M += mask
        weights = F.softmax(M, -1)

        x_att = torch.matmul(weights, v) # (B, S, S) x (B, S, d_model) = (B, S, d_model)
    
        return x_att
 
    def forward(self, query, key, value, mask=None):
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)

        x_att = self.compute_attention(q, k, v, mask)

        return x_att

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


class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(...)

    def forward(self, x):
        embeds = self.embedding(x)
        return embeds

class PositionalEmbedding(nn.Module):
    pass

import torch
from torch import nn
import torch.nn.functional as F
import math

import copy

class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 vocab_size,
                 N_encoders,
                 N_decoders):
        super().__init__()

        copy_mod = copy.deepcopy

        attention = Attention(d_model)
        feedforward = PositionwiseFFN(d_model, h_dim)
        norm = LayerNorm(d_model)

        self.embedding = Embedding(vocab_size, d_model)

        self.encoder = nn.Sequential([
            EncoderBlock(feedforward, attention, norm) for i in range(N_encoders)
        ])

        self.decoder = nn.Sequential([
            DecoderBlock(feedforward, attention, norm) for i in range(N_decoders)
        ])

        self.LM_head = LM_head(d_model, vocab_size)

    def forward(self, x):
        pass


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

# class Encoder(nn.Module):
#     def __init__(self,
#                  N_encoders,
#                  feedforward,
#                  attention,
#                  norm):
#         super().__init__()
        

# class Decoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


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

class Attention(nn.Module):
    def __init__(self,
                 batch_size,
                 seq_len,
                 d_model, 
                 n_heads):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len # max seq. length
        self.d_model = d_model
        self.n_heads = n_heads
        
        if d_model%n_heads != 0:
            raise ValueError('d_model must divide evenly into n_heads')
        
        h_dim = int(d_model/n_heads)

        self.reshape_for_mh = lambda x : x.contiguous().view(batch_size, seq_len, n_heads, h_dim).permute(0, 2, 1, 3).contiguous().view(-1, seq_len, h_dim) # B * n_heads, S, h_dim
        self.undo_reshape_for_mh = lambda x : x.contiguous().view(batch_size, n_heads, seq_len, h_dim).permute(0, 2, 1, 3).contiguous().view(-1, seq_len, d_model) # B, S, h_dim * n_heads = B, S, d_model 

        self.Q = nn.Linear(h_dim, h_dim) # input of size (B*n_heads, S, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
    
    def compute_attention(self, q, k, v, mask):
        # Implment softmax(QK^T/sqrt(d_k))V
        d_k = k.shape[-1]

        M = torch.bmm(q, k.mT)/math.sqrt(d_k) # (B*n_heads, S, h_dim) x (B*n_heads, h_dim, S) = (B*n_heads, S, S)

        # Mask should only be used for decoder 
        if mask: # mask of shape (S, S)
            M += mask
        weights = F.softmax(M, -1)

        x_att = torch.bmm(weights, v) # (B*n_heads, S, S) x (B*n_heads, S, h_dim) = (B*n_heads, S, h_dim)
    
        return x_att
 
    def forward(self, query, key, value, mask=None):
        query = self.reshape_for_mh(query)
        key = self.reshape_for_mh(key)
        value = self.reshape_for_mh(value)

        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)

        x_att = self.compute_attention(q, k, v, mask)

        x_att = self.undo_reshape_for_mh(x_att) 

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
    def __init__(self, 
                 vocab_size,
                 d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embeds = self.embedding(x)
        return embeds
    

class PositionalEmbedding(nn.Module):
    def __init__(self,
                 d_model):
        super().__init__()

        self.pos_encodings = torch.zeros(20000, d_model)
        positions = torch.arange(20000).unsqueeze(-1)
        # Use log for numerical stability
        denom = torch.exp(torch.log(10000) * (torch.arange(0, d_model, 2) / d_model)).unsqueeze(0) 

        self.pos_encodings[::2] = torch.sin(positions/denom) # multiplication better?
        self.pos_encodings[1::2] = torch.cos(positions/denom)

    def forward(self, x):
        x = x + self.pos_encodings[:, :x.size()[1]] # requires grad?
        return x
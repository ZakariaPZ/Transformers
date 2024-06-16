import torch
from torch import nn
import torch.nn.functional as F
import math

from attention import Attention
from layers import Embedding, PositionalEmbedding, PositionwiseFFN, LayerNorm, LM_head

class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 vocab_size,
                 N_encoders,
                 N_decoders):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model) 

        self.encoder = nn.Sequential([
            EncoderBlock(PositionwiseFFN(d_model, h_dim), 
                         Attention(d_model), 
                         PositionwiseFFN(d_model, h_dim)) for i in range(N_encoders)
        ])

        self.decoder = nn.Sequential([
            DecoderBlock(PositionwiseFFN(d_model, h_dim),
                         Attention(d_model),
                         PositionwiseFFN(d_model, h_dim),) for i in range(N_decoders)
        ])

        self.LM_head = LM_head(d_model, vocab_size)

    def embed(self, x):
        return self.positional_embedding(self.embedding(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, z, mask):
        return self.decoder(x, z, mask)

    def forward(self, x, mask):
        x = self.embed(x)
        out = self.decode(x, self.encode(x), mask)
        return self.LM_head(out)   
    

class Encoder(nn.Module):
    def __init__(self, 
                 N_encoders,
                 d_model,
                 vocab_size,
                 h_dim,
                 seq_len,
                 n_heads):
        super().__init__()

        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model)
        self.encoder = nn.ModuleList([
                EncoderBlock(d_model, h_dim, seq_len, n_heads) for _ in range(N_encoders)
            ])

    def forward(self, x):
        z = self.embed(x)
        z = self.pe(z)

        for layer in self.encoder:
            z = layer(z)

        return z
    

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 seq_len,
                 n_heads):
        super().__init__()

        self.attention = Attention(seq_len, d_model, n_heads)
        self.norm = LayerNorm(d_model) 
        self.feedforward = PositionwiseFFN(d_model, h_dim)
    
    def forward(self, x):

        # Apply self-attention
        x = self.norm(x + self.attention(x, x, x))

        # Apply FF layer
        x = self.norm(x + self.feedforward(x))

        return x


class Decoder(nn.Module):
    def __init__(self, 
                 N_decoders,
                 d_model,
                 vocab_size,
                 h_dim,
                 seq_len,
                 n_heads):
        super().__init__()

        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model)
        self.decoder = nn.ModuleList([
                DecoderBlock(d_model, h_dim, seq_len, n_heads) for _ in range(N_decoders)
            ])

    def forward(self, x, z, mask):
        x = self.embed(x)
        x = self.pe(x)

        for layer in self.decoder:
            x = layer(x, z, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 n_heads,
                 seq_len):
        super().__init__()

        self.feedforward = PositionwiseFFN(d_model, h_dim)
        self.attention = Attention(seq_len, d_model, n_heads)
        self.norm = LayerNorm(d_model)

    def forward(self, x, z, mask):

        # Apply self-attention
        x = self.norm(x + self.attention(x, x, x, mask))

        # Apply cross-attention
        x = self.norm(x + self.attention(x, z, z, mask))

        # Apply FF layer
        x = self.norm(x + self.feedforward(x))

        return x

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

        self.embedding = Embedding(vocab_size, d_model)

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

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def forward(self, x):
        pass

class Encoder(nn.Module):
    def __init__(self, 
                 N_encoders,
                 embedding,
                 pe,
                 encoder
                 ):
        super().__init__()

        ## Embedding 
        ## pos encoding 
        ## Encoder -> z 

        self.embed = embedding
        self.pe = pe 


        self.encoder = encoder



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
    

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 seq_len,
                 n_heads):
        super().__init__()

        self.feedforward = PositionwiseFFN(d_model, h_dim)
        self.attention = Attention(seq_len, d_model, n_heads)
        self.norm = LayerNorm(d_model) 
    
    def forward(self, x):

        # Apply self-attention
        x = self.norm(x + self.attention(x, x, x))

        # Apply FF layer
        x = self.norm(x + self.feedforward(x))

        return x

class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 h_dim,
                 n_heads,
                 seq_len,
                 mask = None):
        super().__init__()

        self.feedforward = PositionwiseFFN(d_model, h_dim)
        self.attention = Attention(seq_len, d_model, n_heads)
        self.norm = LayerNorm(d_model)
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
    
def create_attention_mask(shape):
    mask = torch.triu(torch.ones(shape), 1)
    mask = torch.where(mask == 0, 0, -10000)
    return mask

class Attention(nn.Module):
    def __init__(self,
                 seq_len,
                 d_model, 
                 n_heads):
        super().__init__()

        self.seq_len = seq_len # max seq. length
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert d_model%n_heads == 0, 'Embedding dimension d_model must divide evenly into n_heads'
        
        h_dim = int(d_model/n_heads)

        self.Q = nn.Linear(d_model, h_dim*n_heads) # input of size (B, S, d_model)
        self.K = nn.Linear(d_model, h_dim*n_heads)
        self.V = nn.Linear(d_model, h_dim*n_heads)
    
    def reshape_for_mh(self, x):
        n_batches = x.shape[0]
        seq_len = x.shape[1]
        h_dim = int(self.d_model/self.n_heads)


        x = x.contiguous().view(n_batches, seq_len, self.n_heads, h_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(n_batches*self.n_heads, seq_len, h_dim)

        return x

    def undo_reshape_for_mh(self, x):
        n_batches = int(x.shape[0]/self.n_heads)
        seq_len = x.shape[1]
        h_dim = int(self.d_model/self.n_heads)

        x = x.contiguous().view(n_batches, self.n_heads, seq_len, h_dim)

        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(n_batches, seq_len, self.n_heads*h_dim) # B, S, h_dim * n_heads = B, S, d_model 

        return x

    def compute_attention(self, q, k, v, mask):
        # Implment softmax(QK^T/sqrt(d_k))V
        d_k = k.shape[-1]

        M = torch.bmm(q, k.mT)/math.sqrt(d_k) # (B*n_heads, S, h_dim) x (B*n_heads, h_dim, S) = (B*n_heads, S, S)

        # Mask should only be used for decoder 
        if mask != None: # mask of shape (S, S)
            M += mask
        weights = F.softmax(M, -1)

        x_att = torch.bmm(weights, v) # (B*n_heads, S, S) x (B*n_heads, S, h_dim) = (B*n_heads, S, h_dim)
    
        return x_att
 
    def forward(self, query, key, value, mask=None):

        q = self.Q(query) # M is not symmetric because of this...even when query=key=value
        k = self.K(key)
        v = self.V(value) # shape B, S, h_dim*n_heads

        q = self.reshape_for_mh(q)
        k =  self.reshape_for_mh(k)
        v = self.reshape_for_mh(v) # shape B*n_heads, S, h_dim

        x_att = self.compute_attention(q, k, v, mask) # shape B*n_heads, S, h_dim
        x_att = self.undo_reshape_for_mh(x_att) # shape B, S, h_dim*n_heads

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
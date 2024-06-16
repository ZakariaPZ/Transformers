import torch
from torch import nn
import torch.nn.functional as F
import math

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

        # max seq. length
        self.seq_len = seq_len 

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model%n_heads == 0, 'Embedding dimension d_model must divide evenly into n_heads'

        # Each head has dimension d = d_model/n_heads, so that the output
        # of the attention layer is of the same dimension as the input
        h_dim = int(d_model/n_heads)

        # input of size (batch size, sequence length, d_model)
        self.Q = nn.Linear(d_model, h_dim*n_heads) 
        self.K = nn.Linear(d_model, h_dim*n_heads)
        self.V = nn.Linear(d_model, h_dim*n_heads)
    
    def reshape_for_mh(self, X):
        '''
        When computing multi-head attention, ... # TODO
        '''
        n_batches = X.shape[0]
        seq_len = X.shape[1]
        h_dim = int(self.d_model/self.n_heads)

        X = X.contiguous().view(n_batches, seq_len, self.n_heads, h_dim)
        X = X.permute(0, 2, 1, 3)
        X = X.contiguous().view(n_batches*self.n_heads, seq_len, h_dim)

        return X

    def undo_reshape_for_mh(self, X):
        n_batches = int(x.shape[0]/self.n_heads)
        seq_len = X.shape[1]
        h_dim = int(self.d_model/self.n_heads)

        X = X.contiguous().view(n_batches, self.n_heads, seq_len, h_dim)

        X = X.permute(0, 2, 1, 3)

        # batch size, sequence length, h_dim * n_heads = batch size, sequence length, d_model 
        X = X.contiguous().view(n_batches, seq_len, self.n_heads*h_dim) 

        return X

    def compute_attention(self, Q, K, V, mask):
        '''
        Implements the scaled dot-product attention mechanism. 

        X_att = SoftMax(Q * K^T / sqrt(d_k)) * V
        '''
        d_k = K.shape[-1]

        M = torch.bmm(Q, K.mT)/math.sqrt(d_k) # (B*n_heads, S, h_dim) x (B*n_heads, h_dim, S) = (B*n_heads, S, S)

        # Mask should only be used for decoder 
        if mask != None: # mask of shape (S, S)
            M += mask
        weights = F.softmax(M, -1)

        x_att = torch.bmm(weights, V) # (B*n_heads, S, S) x (B*n_heads, S, h_dim) = (B*n_heads, S, h_dim)

        return x_att
 
    def forward(self, query, key, value, mask=None):
        '''
        query, key and value have no context before reaching the first
        attention layer. x_att contains elements which are a linear combination 
        of each element in the original sequence. Their new values reflect 
        '''
        # For each batch, Q, K, V can be computed in parallel
        Q = self.Q(query) # M is not symmetric because of this...even when query=key=value
        K = self.K(key)
        V = self.V(value) # shape B, S, h_dim*n_heads

        Q = self.reshape_for_mh(Q)
        K =  self.reshape_for_mh(K)
        V = self.reshape_for_mh(V) # shape B*n_heads, S, h_dim

        X_att = self.compute_attention(Q, K, V, mask) # shape B*n_heads, S, h_dim
        X_att = self.undo_reshape_for_mh(X_att) # shape B, S, h_dim*n_heads

        return X_att


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
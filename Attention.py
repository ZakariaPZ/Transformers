import torch
from torch import nn
import torch.nn.functional as F
import math

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

        # the input X is of shape (batch_size, sequence_length, d_model), and is
        # projected to Q, K, V of shape (batch_size, sequence_length, h_dim * n_heads)
        # e.g. When computing the queries, Q = X * Wq:
        # (batch_size, sequence_length, d_model) = (batch_size, sequence_length, d_model) * (d_model, d_model),
        # where d_model = h_dim * n_heads
        self.Wq = nn.Linear(d_model, h_dim*n_heads) 
        self.Wk = nn.Linear(d_model, h_dim*n_heads)
        self.Wv = nn.Linear(d_model, h_dim*n_heads)
    
    def reshape_for_mh(self, X):
        '''
        When computing multi-head attention, ... # TODO
        '''
        n_batches = X.shape[0]
        seq_len = X.shape[1]
        h_dim = int(self.d_model / self.n_heads)

        # Combine batches and heads into one dimension, since each head
        # is independent of the rest and can be computed in parallel
        X = X.contiguous().view(n_batches, seq_len, self.n_heads, h_dim)
        X = X.permute(0, 2, 1, 3)
        X = X.contiguous().view(n_batches*self.n_heads, seq_len, h_dim)

        return X

    def undo_reshape_for_mh(self, X):
        n_batches = int(X.shape[0]/self.n_heads)
        seq_len = X.shape[1]
        h_dim = int(self.d_model/self.n_heads)

        # Split the heads back into separate dimensions from the batches
        X = X.contiguous().view(n_batches, self.n_heads, seq_len, h_dim)
        X = X.permute(0, 2, 1, 3)
        X = X.contiguous().view(n_batches, seq_len, self.n_heads*h_dim) 

        return X

    def compute_attention(self, Q, K, V, mask):
        '''
        Implements the scaled dot-product attention mechanism. 

        X_att = SoftMax(Q * K^T / sqrt(d_k)) * V
        '''

        # Scaling factor d_k 
        d_k = K.shape[-1]

        # Argument for SoftMax, Q * K^T / sqrt(d_k)
        # (batch_size * n_heads, sequence_length, h_dim) * (batch_size * n_heads, h_dim, sequence_length) = (batch_size * n_heads, sequence_length, sequence_length)
        M = torch.bmm(Q, K.mT)/math.sqrt(d_k) 

        # Causal mask should only be used for decoder to prevent the model 
        # from attending to future tokens. Shape (sequence_length, sequence_length).
        if mask != None: 
            M += mask
        weights = F.softmax(M, -1)

        # Compute the output of the attention layer, 
        # shape (batch_size * n_heads, sequence_length, sequence_length) * (batch_size * n_heads, sequence_length, h_dim) = (batch_size * n_heads, sequence_length, h_dim)
        X_att = torch.bmm(weights, V)

        return X_att
 
    def forward(self, query, key, value, mask=None):
        '''
        query, key and value have no context before reaching the first
        attention layer. x_att contains elements which are a linear combination 
        of each element in the original sequence. Their new values reflect 
        '''

        # shape (batch_size, sequence_length, h_dim * n_heads) 
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value) 

        # shape (batch_size * n_heads, sequence_length, h_dim) 
        Q = self.reshape_for_mh(Q)
        K =  self.reshape_for_mh(K)
        V = self.reshape_for_mh(V)

        # Shape (batch_size * n_heads, sequence_length, h_dim)
        X_att = self.compute_attention(Q, K, V, mask)

        # Shape (batch_size, sequence_length, h_dim * n_heads) 
        X_att = self.undo_reshape_for_mh(X_att) 

        return X_att
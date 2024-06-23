#################################################
# Goal : To build a translator using transformers
# Approach : Build out individual components in sort of lego blocks then integrate them together, recording shapes along the way
#################################################

import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # The embeddings are multiplied by the sqrt of d_model (Section 3.4)
        self.output = self.embedding(x) * math.sqrt(self.d_model)
        return self.output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout: float ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
       
        # Using sin and cos function to encode the positions (Section 3.5)
        pos_enc = torch.zeros(self.seq_len, self.d_model) #(seq_len, d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)

        # We are calculating in log space for numeric stability
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) 
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0) #(1, seq_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # In pos_enc x.shape[1] is mentioned, so that we can add positional informatio to the embedding uptil that sequence index
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# The below layer norm is similar to the one in pytorch implementation
# https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 1e-5):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias


# Introducing non-linearity by add Feed forward NN (Section 3.3)
class FeedForward(nn.Module):
    def __init__(self, d_model:int,d_ff:int, dropout:float):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.d_ff = d_ff
        self.linear1 = nn.Linear(self.d_model,self.d_ff, bias=True)
        self.linear2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # The paper uses a ReLU activation
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        
        assert self.d_model % self.h == 0, "d_model not divisible by h"

        self.d_h = self.d_model // self.h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    

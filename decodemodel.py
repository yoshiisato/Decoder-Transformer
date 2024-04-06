import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #Creating Matrix Shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        
        #Create 1D tensor
        pos = torch.arange(self.seq_len).unsqueeze(1)
        #Dividend of Positional encoding, base changed to e
        exp = torch.exp(-torch.arange(0, self.d_model, 2) / self.d_model * torch.log(torch.tensor(10000)))

        #Applying sin to even and cos to odd positions. 
        pe[:, 0::2] = torch.sin(pos * exp)
        pe[:, 1::2] = torch.cos(pos * exp)

        pe = pe.unsqueeze(0) #Change to (1, seq_len, d_model) to broadcast PE over multiple batches

        # Registers the pe tensors as buffers, model parameters that are unchanged during backprop
        self.register_buffer('pe', pe)

    def forward(self, x):
        #Changes pe to (1, x.shape[1], d_model) to broadcast onto (batch_size, x.shape[1], d_model)
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False)
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, dropout):
        # building boolean mask
        mask = torch.triu(query, diagonal=1)
        mask = mask == 1
        
        # d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k) #or d_k?
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = nn.Softmax(attention_scores, dim=-1)
        heads = dropout(attention_scores @ value)
        return heads, attention_scores
    
    def forward (self, q, k, v):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)

        heads, attention_scores = self.attention(query, key, value, self.dropout)
        
        heads = heads.transpose(1,2).contiguous().view(heads.shape[0], heads.shape[1], self.h * self.d_k)
        multihead_attention = self.w_o(heads)
        return multihead_attention

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return self.layernorm(self.dropout(sublayer(x)) + x)
        # return x + self.dropout(sublayer(self.norm(x))) FIGURE THIS ONE OUT

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout): #d_ff is intermediate dimension
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
        
class DecoderBlock(nn.Module):
    def __init__(self, features, attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualBlock(features, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, self.dropout))
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x

class LayerNormalization(nn.Module):
    def __init__(self, features, eps: int=10**-3):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #Features is most likely d_model
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x has dim (batch, seq_len, features)
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        variance = x.var(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)

        return self.alpha * (x - mean) / std + self.bias
    
class Decoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList): #self.layers is composed of a stack of decoderblock layers
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.projection(x)
    
class Transformer(nn.Module):
    def __init__(self, src_embed: InputEmbedding, pos_enc: PositionalEncoding, decoder: Decoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.pos_enc = pos_enc
        self.decoder = decoder
        self.projection_layer = projection_layer

    def forward(self, x):
        x = self.src_embed(x)
        x = self.pos_enc(x)
        x = self.decoder(x)
        x = self.projection_layer(x)
        return x

# Decoder is composed of a stack of N = 6 identical layers, and each multi-head attention block is composed of h = 8 parallel attention layers (heads)
def build_transformer(vocab_size: int, seq_len: int, d_model: int=512, N_x: int=6, h: int=8, d_ff: int=2048, dropout: float=0.1):
    # create input embeddings
    word_embedding = InputEmbedding(d_model, vocab_size)

    # create positional encoding
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

    decoder_blocks = []
    for _ in range(N_x):
        self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, vocab_size)

    transformer = Transformer(word_embedding, positional_encoding, decoder, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
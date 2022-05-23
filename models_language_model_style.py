import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import copy


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_encoder = nn.Embedding(max_len, d_model)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.pos_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, pos_id_torch_pad=None):
        if pos_id_torch_pad is None:
            pe = self.pos_encoder.weight.data
            pe = pe.unsqueeze(0).transpose(0, 1)
            x = x + pe[:x.size(0), :]
        else:
            # assigned pos_encoding
            x = x + self.pos_encoder(pos_id_torch_pad)
        return self.dropout(x)


class PositionalEncodingWithoutDropout(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingWithoutDropout, self).__init__()

        self.pos_encoder = nn.Embedding(max_len, d_model)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.pos_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, pos_id_torch_pad=None):
        if pos_id_torch_pad is None:
            pe = self.pos_encoder.weight.data
            pe = pe.unsqueeze(0).transpose(0, 1)
            x = x + pe[:x.size(0), :]
        else:
            # assigned pos_encoding
            x = x + self.pos_encoder(pos_id_torch_pad)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, src_mask=None, key_padding_mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, src_mask, key_padding_mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, src_mask=None, key_padding_mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if src_mask is not None:
        if src_mask.dtype == torch.bool:
            scores.masked_fill(src_mask.unsqueeze(1), float('-inf'))
        else:
            scores += src_mask.unsqueeze(0).unsqueeze(1)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=50, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask, key_padding_mask=None):
        x = x + self.attn(self.norm_1(x), x, x, src_mask, key_padding_mask)
        x = self.norm_1(x)

        x = x + self.dropout_1(self.ff(x))

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.2, norm=None):
        super().__init__()
        self.N = N

        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = norm

    def forward(self, src, src_mask, key_padding_mask=None):

        for i in range(self.N):
            src = self.layers[i](src, src_mask, key_padding_mask)
        if self.norm is not None:
            src = self.norm(src)
        return src


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerNP5Model(nn.Module):
    def __init__(self, vocab_size, d_model, nlayers, nheads, ninterests=3, dropout=0.2):
        super().__init__()
        self.se = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.transformer_encoder_1 = Encoder(d_model, 1, nheads, dropout=dropout, norm=nn.LayerNorm(d_model))
        self.transformer_encoder_2 = get_clones(Encoder(d_model, nlayers, nheads, dropout=dropout, norm=nn.LayerNorm(d_model)), ninterests)
        self.transformer_encoder_3 = Encoder(d_model, 1, nheads, dropout=dropout, norm=nn.LayerNorm(d_model))
        self.d_model = d_model
        self.ninterest = ninterests

        self.C1 = nn.Linear(d_model, ninterests)
        self.C2 = nn.Softmax(dim=-1)

    def forward(self, src, src_mask, key_padding_mask=None, src_pos_id=None):

        self.src_emb = self.se(src)
        src_emb_pe = self.pe(self.src_emb, src_pos_id)
        src_1 = self.transformer_encoder_1(src_emb_pe, src_mask, key_padding_mask)
        self.src_for_c = src_1.detach()

        c = self.transformer_encoder_3(self.src_for_c, src_mask, key_padding_mask)
        c = self.C1(c)  # /0.001 # divide temperature
        self.activ_matrix = self.C2(c).unsqueeze(-1)  # [batch_size, seq_len, interest_num, 1]

        # if self.training:
        #     m = torch.distributions.dirichlet.Dirichlet(0.5 * torch.ones_like(self.activ_matrix.squeeze(-1)))
        #     diri_noise = m.sample()
        #     self.activ_matrix = 0.75 * self.activ_matrix + 0.25 * diri_noise.unsqueeze(-1)

        self.src_interest = torch.matmul(self.activ_matrix, src_1.unsqueeze(-2))
        output_list = []

        for i in range(self.ninterest):
            src_interest = self.src_interest[:, :, i, :]
            # src_interest = self.pe(src_interest, src_pos_id)
            output = self.transformer_encoder_2[i](src_interest, src_mask, key_padding_mask)

            output_list.append(output)

        return torch.stack(output_list, dim=-2)


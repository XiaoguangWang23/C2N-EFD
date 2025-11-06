import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from math import sqrt




class MLP(nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dims):
        super().__init__()
        self.mlp_layers = [mlp_input_dim] + mlp_hidden_dims
        self.mlp_layer_list = []
        for layer in list(zip(self.mlp_layers[:-1], self.mlp_layers[1:])):
            self.mlp_layer_list.append(nn.Linear(layer[0], layer[1]))
            self.mlp_layer_list.append(nn.BatchNorm1d(layer[1]))
            self.mlp_layer_list.append(nn.LeakyReLU())
            self.mlp_layer_list.append(nn.Dropout())

        self.mlp_layer_list.append(nn.Linear(self.mlp_layers[-1], 1))
        self.mlp_layer_list.append(nn.Sigmoid())
        self._mlp = nn.Sequential(*self.mlp_layer_list)

    def forward(self, x):
        out = self._mlp(x)
        return out


class Embedder(nn.Module):
    def __init__(self, cat_cards, d):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in cat_cards])
    def forward(self, cx):
        cat_vecs = torch.stack([emb(cx[:,i]) for i,emb in enumerate(self.cat_emb)], 1)  # [B,Nc,d]
        return cat_vecs


class C2D(nn.Module):
    """
    Intra-feature attention: query = embedding of selected category value;
    keys/values = all category embeddings of that feature.
    Output: scalar per categorical feature.
    """
    def __init__(self, d, cat_cards, hidden_mult=2):
        super().__init__()
        self.d = d
        self.Nc = len(cat_cards)
        self.Wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.Nc)])
        self.Wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.Nc)])
        self.Wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.Nc)])
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden_mult*d), nn.ReLU(), nn.Linear(hidden_mult*d, d)) for _ in range(self.Nc) ])
        self.ln2= nn.LayerNorm(d)

        self.to_scalar = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 1), nn.Sigmoid()) for _ in range(self.Nc)
        ])

        # self.to_scalar = nn.ModuleList([nn.Linear(d,1) for _ in range(self.Nc)])
    def forward(self, cx, cat_vecs, embed_weights):
        # cx: [B,Nc] indices, cat_vecs: [B,Nc,d], embed_weights: list of (Ci,d)
        scalars=[]
        for j in range(self.Nc):
            q = self.Wq[j](cat_vecs[:,j])                       # [B,d]
            k_all = self.Wk[j](embed_weights[j])                 # [Ci,d]
            v_all = self.Wv[j](embed_weights[j])              # [Ci,d]
            attn_scores = torch.matmul(q, k_all.t()) / sqrt(self.d)   # [B,Ci]
            alpha = torch.softmax(attn_scores, dim=-1)         # [B,Ci]
            h = torch.matmul(alpha, v_all)                  # [B,d]
            h = self.ln1(cat_vecs[:,j] + h)                 # Add&Norm
            h = self.ln2(h + self.ff[j](h))                 # FF + Add&Norm
            scalars.append(self.to_scalar[j](h).squeeze(-1))
        return torch.stack(scalars,1)


class Explicit(nn.Module):
    def __init__(self, dim_in, cross_layers=1):
        super().__init__()
        self.cross_layers = cross_layers
        self.linears = nn.ModuleList([nn.Linear(dim_in, dim_in, bias=False) for i in range(self.cross_layers)])

    def forward(self, x):
        explicit_feats = []
        x0 = x
        xi = x
        explicit_feats.append(x0)
        for i in range(self.cross_layers):
            xi = x0 * self.linears[i](xi)
            explicit_feats.append(xi)
        return explicit_feats


class EFD(nn.Module):
    def __init__(self, cat_cards, Nc, Nu, d=8, cross_layers=1, mlp_hidden_dims=[256,16]):
        super().__init__()
        self.embed = Embedder(cat_cards, d)
        self.c2d = C2D(d, cat_cards)
        self.expl = Explicit(dim_in=Nc+Nu, cross_layers=cross_layers)
        explicit_dim = (Nc+Nu)*(cross_layers+1)
        fusion_dim =  explicit_dim
        self.mlp = MLP(mlp_input_dim=fusion_dim,mlp_hidden_dims=mlp_hidden_dims)
    def forward(self, cx, nx):
        cat_vecs = self.embed(cx)
        c2d_scalar = self.c2d(cx, cat_vecs, [emb.weight for emb in self.embed.cat_emb])
        explicit_in = torch.cat([nx, c2d_scalar],1)
        explicit_out = self.expl(explicit_in)
        fusion = torch.cat(explicit_out, -1)
        return self.mlp(fusion).squeeze(-1),c2d_scalar
















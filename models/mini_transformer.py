import torch
import torch.nn as nn
import torch.utils.data as data
import math
from pytorch_model_summary import summary

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AccelerometerModel(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        # self.emb = nn.Embedding(4,dim)
        self.emb = nn.Conv1d(3, 4, 5, 5) # 
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        # self.positional_encoding = PositionalEncoding(dim, 900)
        self.conv2 = nn.Conv1d(4, 1, 5, 5)
        self.proj_out = nn.Linear(360,3)
    
    def forward(self, x0):
        # mask = x0['mask']
        # Lmax = mask.sum(-1).max()
        # mask = mask[:,:Lmax]
        # x = x0['seq'][:,:Lmax]
        x = x0
        
        pos = torch.arange(1800, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos).transpose(2, 1)
        x = self.emb(x)
        # print(x.size(), pos.size())
        x = x + pos
        # x = self.positional_encoding(x)
        # x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.transformer(x.transpose(2, 1))
        x = self.conv2(x.transpose(2, 1))
        # print(x.size())
        x = x.view(x.size(0), 360)
        x = self.proj_out(x)
        
        return x

if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    transformer = AccelerometerModel(dim=4, head_size=1)
    data = torch.randn(8, 3, 900)
    output = transformer(data)
    print(output.size())
    # # show input shape
    print(summary(transformer, data, show_input=True))

    # # show output shape
    # print(summary(transformer, data, show_input=False))

    # show output shape and hierarchical view of net
    # print(summary(transformer, *(src_data, tgt_data), show_input=False, show_hierarchical=True))
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model % num_heads == 0 
        self.d_model = d_model 
        self.d_head = d_model // num_heads
        self.num_heads = num_heads 
        self.qproj = nn.Linear(d_model,d_model,bias = False)
        self.kproj = nn.Linear(d_model,d_model,bias = False)
        self.vproj = nn.Linear(d_model,d_model,bias = False)
        self.outproj = nn.Linear(d_model,d_model,bias = False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        
        B,T,d_model = x.shape
        Q = self.qproj(x)
        K = self.kproj(x)
        V = self.vproj(x)

        Q = Q.view(B,T,self.num_heads,self.d_head)
        K = K.view(B,T,self.num_heads,self.d_head)
        V = V.view(B,T, self.num_heads,self.d_head)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        dot_product = Q @ K.transpose(-2,-1)
        score = dot_product/(self.d_head ** 0.5)

        if mask is not None: 
            score = score.masked_fill(mask == 0,float('-inf')) 
        
        softmax_output =  torch.softmax(score,dim = -1)
        attention = softmax_output  @ V 
        attention = attention.transpose(1,2)

        attention = attention.contiguous().view(B,T,self.d_model)

        output = self.outproj(attention)
        return output 

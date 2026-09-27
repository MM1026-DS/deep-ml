import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp, dropout 
        self.d_model  = d_model 
        self.num_heads = num_heads 
        self.d_ff = d_ff 
        self.dropout = nn.Dropout(dropout) 
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model,self.num_heads,dropout=self.dropout,batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
                        nn.Linear(self.d_model,self.d_ff), 
                        nn.GELU(), 
                        nn.Linear(d_ff,d_model)
        )
        



        

    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        xnorm = self.norm1(x)
        attentionx,_ = self.attention(xnorm,xnorm,xnorm)
        x = x + self.dropout(attentionx)

        xnorm = self.norm2(x)
        mlpoutput = self.mlp(xnorm)
        x = x + self.dropout(mlpoutput)
        return x

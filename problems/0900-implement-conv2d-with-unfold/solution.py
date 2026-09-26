import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
   B, Cin, Hin, Win = x.shape 
   Cout,_ , kh, kw = W.shape
   patches = F.unfold(
              x, 
              kernel_size = (kh,kw),
              stride = stride, 
              padding = padding)
   W = W.reshape(Cout, W.shape[1]*kh*kw)

   y = torch.matmul(
        W,
        patches
    )
   if b is not None:
    y = y + b.view(1, -1, 1)

   Hout = ((Hin +2*padding - kh)// stride) +1  
   Wout = ((Win + 2*padding - kw)// stride ) +1 

   return y.reshape(B,Cout,Hout,Wout)


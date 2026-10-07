import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: create W_ih, W_hh, b_ih, b_hh with the shapes specified above
        self.W_ih = nn.Parameter(torch.zeros(4*hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.zeros(4*hidden_size,hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4*hidden_size,))
        self.b_hh = nn.Parameter(torch.zeros(4*hidden_size,))
        

    def forward(self, x, state):
        h_prev, c_prev = state
        # TODO: compute the four gates and return (h_new, c_new)
        gates = x @ self.W_ih.T + self.b_ih + h_prev @ self.W_hh.T + self.b_hh

        ## split inot four chunks 
        i,f,g,o = gates.chunk(4,dim = 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = c_prev * f + i*g
        h_new = o * torch.tanh(c_new)

        

        # ## information store 
        # gt = nn.Tanh(x @ self.W_in.T + self.b_in + h_prev @ self.W_hh + self.b_hh)

        # ## add gate 
        # at = nn.Sigmoid(x @ self.W_in.T + self.b_in + h_prev @ self.W_hh + self.b_hh)
        # jt = at * gt 

        # cnew = jt + kt 

        # ## output gate 
        # ot = nn.Sigmoid(x @ self.W_in.T + self.b_in + h_prev @ self.W_hh + self.b_hh)
        # hnew = ot * nn.Tanh(cnew)


        return h_new,c_new

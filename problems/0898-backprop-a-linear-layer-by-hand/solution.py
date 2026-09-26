import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    # dl/dx = dl/dy * dy/dx = (grad_output @ W) 
    ## dl/dw = (grad_output @ X)
    ## dl/db = dl/dw 

    return grad_output @ W, grad_output.t() @ x , grad_output.sum(dim = 0)
import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    x = x.flatten() 
    y = x.reshape(new_shape)
    return y 

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    # TODO: swap the last two dimensions of x
    y = None 
    if len(x.shape)<3:
        y = x.t()
    else:
        y = x.permute(0,2,1)
    return y 

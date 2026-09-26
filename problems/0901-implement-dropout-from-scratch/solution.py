import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    ot = None 
    if training:
        mask = ((torch.rand_like(x)>p).to(x.dtype))
        ot = x * mask/(1-p)
    else:
        ot = x
    return ot 


import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    PE = torch.zeros(seq_len,d_model)

    for p in range(seq_len):
        for j in range(d_model):
            i = j//2 

            den = math.exp( - math.log(10000) * (2*i)/d_model)
            if j%2 ==0:
                PE[p,j] = math.sin(p*den)
            else:
                PE[p,j] = math.cos(p*den) 

    return PE

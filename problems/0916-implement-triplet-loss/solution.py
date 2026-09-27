import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    ## l = max(0,||anchor - pos||**2 - ||anchor - neg||**2 + margin)
    # loss =nn.MSELoss() 

    l1 = torch.sum((anchor - positive)**2,dim = -1)
    l2 = torch.sum((anchor - negative)**2, dim=-1) 

   
    total_loss = torch.clamp((l1-l2 + margin), min = 0)
    return total_loss.mean()
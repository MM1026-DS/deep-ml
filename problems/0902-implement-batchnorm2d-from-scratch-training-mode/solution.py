import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    
    mean = x.mean(dim = (0,2,3),keepdim = True)

    var = x.var(
         dim = (0,2,3), 
         keepdim = True, 
         unbiased = False
    )

    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)

    xout = (x - mean)/torch.sqrt(var + eps)

    yout = xout * gamma + beta 
    return yout 
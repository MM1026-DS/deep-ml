import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        ctx.save_for_backward(x)
        return torch.clamp(x,min = 0)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        x, = ctx.saved_tensors
        grad_x = grad_output.clone() 

        grad_x[x<=0] = 0
        return grad_x 

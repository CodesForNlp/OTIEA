import torch
def Test(model):
    model.eval()
    with torch.no_grad:
        x1=0
        x2=None
    return x1,x2
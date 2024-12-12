import torch
import torch.nn as nn
import torch.nn.functional as F


class CauchyKernel(nn.Module):

    def __init__(self, scale=1.):
        super(CauchyKernel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale]))

    def forward(self, x, y):
        dist = torch.squeeze(torch.cdist(torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)), 0)

        # Following authors' decision, scale is transformed into theta via softplus
        return 1 / (1 + dist / torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))


class ExpQuadKernel(nn.Module):
    
    def __init__(self, scale=1.):
        super(ExpQuadKernel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale]))

    def forward(self, x, y):
        dist = torch.squeeze(torch.cdist(torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)), 0)

        # Following authors' decision, scale is transformed into theta via softplus
        return torch.exp(-dist / torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
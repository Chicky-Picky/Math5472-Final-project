import torch
import torch.nn.functional as F

def NBLoss(y, mu, theta, scale_factor=None):
    eps = 1e-10
    
    if scale_factor is not None:
        scale_factor = scale_factor[:, None]
        mu *= scale_factor

    return (torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(y + theta + eps) +
            (theta + y) * torch.log(1.0 + mu / (theta + eps)) + y * (torch.log(theta + eps) - torch.log(mu + eps))).sum()

def CE(mu_1, sigma2_1, mu_2, sigma2_2):
    return (-(1.8378770664093453 + torch.log(sigma2_2) +
              (sigma2_1 + mu_1 ** 2 - 2 * mu_1 * mu_2 + mu_2 ** 2) / sigma2_2) / 2)

def ELBO(output):
    ELBO = output[0]

    return ELBO

def NBMetric(output):
    NB = output[1]
        
    return NB.item() / output[-2].shape[0]

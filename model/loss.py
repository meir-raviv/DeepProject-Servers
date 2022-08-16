import torch.nn.functional as F
import torch

def l1_loss(output, target, vec=None, weight=1):
    #return (torch.abs(target - output))
    return torch.mean(torch.abs(target[:, 0] - (output[:, 0] + output[:, 1]))) #weight * 

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_loss(output, target, weight):
    return F.binary_cross_entropy(output, target, weight=weight)

def nll_loss(output, target):
    return F.nll_loss(output, target)
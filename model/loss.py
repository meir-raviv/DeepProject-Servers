import torch.nn.functional as F
import torch

def l1_loss(output, target, weight=1, vec=None):
    return (torch.abs(target - output))
    #print(target[:, 0])
    #print(output[:, 1].shape)
    #return torch.mean(torch.sum(weight[:, 0] * torch.abs(target[:, 0] - (output[:, 0] + output[:, 1])), dim=(1, 2, 3))) #weight * 

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_loss(output, target, weight):
    return F.binary_cross_entropy(output, target, weight=weight)

def nll_loss(output, target):
    return F.nll_loss(output, target)
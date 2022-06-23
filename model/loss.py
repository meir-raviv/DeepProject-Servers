import torch.nn.functional as F


def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_loss(output, target, weight):
    return F.binary_cross_entropy(output, target, weight=weight)

def nll_loss(output, target):
    return F.nll_loss(output, target)
from torch import nn
import torchvision

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101).__init__()
        self.RN101 = torchvision.models.ResNet101(pretrained=True)
        for param in self.RN101.parameters():
            param.requires_grad = False

    def forward(self, X):
        pass
        

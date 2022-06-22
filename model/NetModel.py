from torch import nn
from AudioVisualSeparator import AudioVisualSeparator
from ResNet101 import ResNet101

class CoSeparationNet(nn.Module):
    def __init__(self):
        super(CoSeparationNet, self).__init__()
        self.RN101 = ResNet101
        self.AVS1 = AudioVisualSeparator()
        self.AVS2 = AudioVisualSeparator()
        self.AVS3 = AudioVisualSeparator()

    def forward(self, X):


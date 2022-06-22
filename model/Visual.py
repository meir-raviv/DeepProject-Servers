import torch
import torchvision
from .ResNet18 import ResNet18

class Visual:
    #visual case
    def create_visual_vector(self, input_channel=3, fc_out=512, pretrained_weights=''):
        
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet = ResNet18(resnet, pool_type="conv1x1", input_channel=3, with_fc=True, fc_in=6272, fc_out=fc_out)

        # if pretrained_weights != '':
        #     print('using pretrained weights for visual')
        #     resnet.load_state_dict(torch.load(pretrained_weights))

        return resnet
import torch
import torchvision
from .ResNet18 import ResNet18

class Classifier:
    #classifier case - for weak loss
    def get_audio_classification(self, input_channel=3, fc_out=15, pretrained_weights='', class_amount=9):
        
        resnet = torchvision.models.resnet18(True)
        resnet = ResNet18(resnet, pool_type="maxpool", input_channel=1, with_fc=True, fc_in=512, fc_out=class_amount)

        # if pretrained_weights != '':
        #     print('using pretrained weights for audio classification')
        #     resnet.load_state_dict(torch.load(pretrained_weights))
        return resnet
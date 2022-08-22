from torch import nn
import torchvision
import torch
import torch.nn.functional as F

class ResNet18(nn.Module):

    def __init__(self, original_resnet, pool_type, input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(ResNet18, self).__init__()
        
        # self.RN18 = torchvision.models.ResNet18(pretrained=True)
        # for param in self.RN18.parameters():
        #     param.requires_grad = False
            
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        # since the ResNet18 should handle both visual and audio, we need to adjust the first conv layer accordingly, distinguished 
        # using the "self.input_layer". this first layer will replace the first original ResNet18's layer. Similarly, the last 2 layers
        # of the original Resnet18 are a fc layer and a softmax layer, which we are changing to our own fc layer, which its size also 
        # depends on whether we are dealing with visual or audio:
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers_list = [self.conv1]
        layers_list.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers_list) #features before pooling
        

        if pool_type == 'conv1x1': # meaning visual 
            self.conv1x1 = self.create_convolution_layer(512, 128, 1, 0) # from 512*7*7 to 128*7*7

            #apply weigts...?


        if with_fc: # regarding visual and audio as well
            self.fc = nn.Linear(fc_in, fc_out)

    def forward(self, x):
        x = self.feature_extraction(x)#[128, 512, 7, 7]
        #print("conv1x1")
        #print(x.shape)

        
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1) #[128, 512, 1, 1]
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        elif self.pool_type == 'conv1x1':
            x = self.conv1x1(x)
        else:
            return x
        #print("fc")
        #print(x.shape)
        if self.with_fc:
            x = x.view(x.size(0), -1) # we want to flatten the vector, x.size(0) holds the number of pictures/spectograms
            x = self.fc(x)
            if self.pool_type == 'conv1x1':
                x = x.view(x.size(0), -1, 1, 1) # the visual tensor needs to have 4 dimensions so we go back from 2 to 4
            return x
        else:
            return x
    
    #add a convolution layer in the end of the model (for the visual case)
    def create_convolution_layer(self, in_chnl, out_chnl, krnl, pad, batch_norm=True, Relu=True, stride=1):
        layer = [nn.Conv2d(in_chnl, out_chnl, krnl, stride=stride, padding=pad)]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_chnl))
        if Relu:
            layer.append(nn.ReLU())
        
        return nn.Sequential(*layer)
        
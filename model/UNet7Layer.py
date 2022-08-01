from torch import nn
import torch
import torchvision

class UNet7Layer(nn.Module):
    def __init__(self, input=2, next_layer_frames=64, output=1):
        super(UNet7Layer, self).__init__()
        self.down_layer1 = nn.Sequential(*[nn.Conv2d(in_channels=input, out_channels=next_layer_frames, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer2 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames, out_channels=next_layer_frames*2, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*2),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer3 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames*2, out_channels=next_layer_frames*4, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*4),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer4 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames*4, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer5 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames*8, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer6 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames*8, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.LeakyReLU(0.2, True)])
        self.down_layer7 = nn.Sequential(*[nn.Conv2d(in_channels=next_layer_frames*8, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.LeakyReLU(0.2, True)])

        '''Here (in the forward section) we Concatenate The ResNet18 Visual Vector so frames are doubled, which will increase the number of
        channels from 512*2*2 to 1024*2*2'''

        self.up_layer1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*16, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.ReLU(True)])
        self.up_layer2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*16, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.ReLU(True)])
        self.up_layer3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*16, out_channels=next_layer_frames*8, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*8),
                                      nn.ReLU(True)])
        self.up_layer4 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*16, out_channels=next_layer_frames*4, kernel_size=4, stride=2, padding=1),                                      
                                      nn.BatchNorm2d(next_layer_frames*4),
                                      nn.ReLU(True)])
        self.up_layer5 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*8, out_channels=next_layer_frames*2, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames*2),
                                      nn.ReLU(True)])
        self.up_layer6 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*4, out_channels=next_layer_frames, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(next_layer_frames),
                                      nn.ReLU(True)])
        self.up_layer7 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=next_layer_frames*2, out_channels=output, kernel_size=4, stride=2, padding=1),
                                      nn.Sigmoid()])

    def forward(self, audio_in, visual_in):
        down_1 = self.down_layer1(audio_in)
        down_2 = self.down_layer2(down_1)
        down_3 = self.down_layer3(down_2)
        down_4 = self.down_layer4(down_3)
        down_5 = self.down_layer5(down_4)
        down_6 = self.down_layer6(down_5)
        down_7 = self.down_layer7(down_6)

        #print("----")
        #print(visual_in.shape)
        
        '''Adjust The Visual Vector To Be The Size Of The U-Net 7th Layer Vector:
        the output of the Resnet18 will be a 512 vector, we will replicate it 2*2 times to create a 512*2*2 block'''
        visual_in = visual_in.repeat(1, 1, down_7.shape[2], down_7.shape[3]) # down_7.shape[2] = 2, down_7.shape[3] = 2
        '''check this command on a stub vector'''

        #print(visual_in.shape)
        
        #print("----")
        #print(down_7.shape)

        v = torch.cat((visual_in, down_7), dim=1)
        up_1 = self.up_layer1(v) # here we concatenate the visual 512*2*2 block
        # to the 512*2*2 output of the down sampling to create a 1024*2*2 block
        up_2 = self.up_layer2(torch.cat((up_1, down_6), dim=1))
        up_3 = self.up_layer3(torch.cat((up_2, down_5), dim=1))
        up_4 = self.up_layer4(torch.cat((up_3, down_4), dim=1))
        up_5 = self.up_layer5(torch.cat((up_4, down_3), dim=1))
        up_6 = self.up_layer6(torch.cat((up_5, down_2), dim=1))
        up_7 = self.up_layer7(torch.cat((up_6, down_1), dim=1))

        '''check if down and up are symmetrical and if not, use crop for skip connections'''
        mask = up_7
        return mask

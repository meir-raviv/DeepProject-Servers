import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch
from .Visual import Visual
from .UNet7Layer import UNet7Layer
from .Classifier import Classifier
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np
import librosa

import matplotlib.pyplot as plt

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
        
class AudioVisualSeparator(nn.Module):
    def __init__(self):
        super(AudioVisualSeparator, self).__init__()
        self.visual = Visual().create_visual_vector()
        self.uNet7Layer = UNet7Layer(input=1)
        self.classifier = Classifier()  #for weak labels
    
    def device(self, dev):
        self.device = dev
        
    def plot_spectrogram(self, spec, id, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db)')
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        plt.show(block=False)
        path = r'/home/dsi/ravivme/run-model/Servers-DeepProject/DeepProject-Servers/specs/spec_' + id
        plt.savefig(path + '.png')
        
    '''X is the input and will in a format of a dictionary with several entries'''
    def forward(self, X):
        vid_ids = X['ids'].view(-1)           # + [X['obj2']['id']]
        bs = X["detections"].shape[0]
        audio_mags = X['audio_mags'].view(bs, 1, 512, 256)            #['stft'][0], X['obj2']['audio']['stft'][0]]  #array includes both videos data - 2 values
        mixed_audio = X['mixed_audio'].view(bs, 1, 512, 256) + 1e-10
        detected_objects = X['detections']
        classes = X['classes'].view(-1) - 1

        # for idx, _ in enumerate(classes):
        #     classes[idx] -= 1
            
        # for idx1, obj in enumerate(classes):
        #     for idx2, _ in enumerate(obj):
        #         if classes[idx1][idx2] == -2:
        #             classes[idx1][idx2] = 15


          # warp the spectrogram
        B = mixed_audio.size(0)     # * mixed_audio.size(1)
        T = mixed_audio.size(3)
        
        grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.device)
        #mixed_audio_simple = mixed_audio[:, 0]
        mixed_audio = F.grid_sample(mixed_audio, grid_warp)
        audio_mags = F.grid_sample(audio_mags, grid_warp)

        log_mixed_audio = torch.log(mixed_audio).detach()
        log_mixed_audio = log_mixed_audio.view(bs, 1, 256, 256)

        ''' mixed audio and audio are after STFT '''
        
        # mask for the object
        ground_mask = audio_mags / mixed_audio     #list of masks per video
        #should we clamp ? -
        ground_mask = ground_mask.clamp(0, 5)
        #ground_mask = F.grid_sample(ground_mask, grid_warp)

        # self.plot_spectrogram(audio_mags[0][0], "audio_" + str(vid_ids[0]))
        # self.plot_spectrogram(audio_mags[2][0], "audio_" + str(vid_ids[2]))
        # self.plot_spectrogram(ground_mask[0][0] + ground_mask[2][0], "mask_" + str(vid_ids[0]))
        # self.plot_spectrogram(mixed_audio[0][0], "mix_" + str(vid_ids[0]))
        
        #for i, mask in enumerate(ground_mask):
        #    self.plot_spectrogram(mask[0], str(vid_ids[i]))

        # Resnet18 for the visual part of the detected object
        #print(detected_objects.shape)
        visual_vecs = self.visual(Variable(detected_objects, requires_grad=False))

        mask_preds = self.uNet7Layer(log_mixed_audio, visual_vecs)

        masks_applied = mixed_audio * mask_preds   #   torch.from_numpy(mixed_audio) * mask_preds

        # after this there will be an iSTFT in the next layer of the net if we would like to hear the sound...

        spectro = torch.log(masks_applied + 1e-10)#.view(bs, 1, 256, 256)  # get log spectrogram

        # loss weights
        weights = torch.log1p(mixed_audio)
        weights = torch.clamp(weights, 1e-3, 10)

        audio_label_preds = self.classifier.get_audio_classification().forward(spectro)
        

        return {"ground_masks" : ground_mask, "ground_labels" : classes, "predicted_audio_labels" : audio_label_preds,
                "predicted_masks" : mask_preds, "predicted_spectrograms" : masks_applied,
                "visual_objects" : visual_vecs, "mixed_audios" : mixed_audio, "videos" : vid_ids, "weights" : weights}

'''https://github.com/rhgao/co-separation/blob/bd4f4fd51f2d6090e1566d20d4e0d0c8d83dd842/models/audioVisual_model.py'''

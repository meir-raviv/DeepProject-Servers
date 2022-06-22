from torch import nn
import torch
from .Visual import Visual
from .UNet7Layer import UNet7Layer
from .Classifier import Classifier
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np


'''
->  each object X has 2 dictionaries, each for a single clip with the following structure:
    {
        'id'     :   video id number
        'audio'  :   { 'wave' : (wave, sr), 'stft' : (mags, phases) }
        'images' :   [(class_id1, image), (class_id2, image),...] -> just 1 or 2 cropped images
    }
    
->  and total:
    
    {
        "obj1": obj1,
        "obj2": obj2,
        "mix": (mix_mags, mix_phases)
    }
'''

    
class AudioVisualSeparator(nn.Module):
    def __init__(self, log):
        super(AudioVisualSeparator, self).__init__()
        self.visual = Visual().create_visual_vector()
        self.uNet7Layer = UNet7Layer(input=1)
        self.classifier = Classifier()  #for weak labels
        

    '''X is the input and will in a format of a dictionary with several entries'''
    
    def forward(self, X):
        vid_ids = X['ids']           # + [X['obj2']['id']]
        audio_mags = X['audio_mags']            #['stft'][0], X['obj2']['audio']['stft'][0]]  #array includes both videos data - 2 values
        mixed_audio = X['mixed_audio']
        detected_objects = torch.from_numpy(X['detections'])
        classes = X['classes']

        log_mixed_audio = torch.log(torch.from_numpy(mixed_audio)).detach()
        self.log.write(log_mixed_audio.shape)
        log_mixed_audio = log_mixed_audio.view(128, 1, 512, 256)
        
        '''mixed audio and audio are after STFT '''
             
        # mask for the object
        ground_mask = audio_mags / mixed_audio     #list of masks per video
        #should we clamp ? -
        ground_mask = torch.from_numpy(ground_mask).clamp(0, 5)

        # Resnet18 for the visual part of the detected object
        visual_vecs = self.visual(Variable(detected_objects, requires_grad=False))
        #visual_vecs = visual_vecs.view(128, 1, 512, 256)

        mask_preds = self.uNet7Layer(log_mixed_audio, visual_vecs)

        masks_applied = torch.from_numpy(mixed_audio) * mask_preds

        # after this there will be an iSTFT in the next layer of the net if we would like to hear the sound...

        spectro = torch.log(masks_applied + 1e-10)  # get log spectrogram

        audio_label_preds = self.classifier.get_audio_classification(spectro)

        return {"ground_masks" : ground_mask, "ground_labels" : classes, "predicted_audio_labels" : audio_label_preds,
                "predicted_masks" : mask_preds, "predicted_spectrograms" : masks_applied,
                "visual_objects" : visual_vecs, "mixed_audios" : mixed_audio, "videos" : vid_ids}
        
'''https://github.com/rhgao/co-separation/blob/bd4f4fd51f2d6090e1566d20d4e0d0c8d83dd842/models/audioVisual_model.py'''




from torch.utils.data import Dataset
import os
import random
import librosa
from torchvision.io import read_image
import torchvision.transforms as T
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


'''
classes_dict = {'Banjo':1, 'Cello':2, 'Drum':3, 'Guitar':4,
                       'Harp':5, 'Harmonica':6, 'Oboe':7, 'clarinet':7, 'Piano':8, 'xylophone':8, 'Saxophone':9,
                       'Trombone':10, 'Trumpet':11, 'Violin':12, 'Flute':13,
                       'Accordion':14, 'Horn':15, 'tuba':15, 'erhu':99}
'''

# the files in data_dir will be enumerated from 000000, and contain pickles objects
class MusicDataset(Dataset):

    def __init__(self, data_dir, transform, log=None, train=True):
        self.log = log
        if self.log is None:
            try:
                self.log = open(r"/dsi/gannot-lab/datasets/Music/Logs/RunLog.txt", "x")
            except:
                self.log = open(r"/dsi/gannot-lab/datasets/Music/Logs/RunLog.txt", "w")

        self.dir_path = data_dir
        self.transform = transform
        self.size = 0

        try:
            self.size = len(os.listdir(self.dir_path))
        except OSError:
            self.log.write(" -->> " + self.dir_path + " is not a valid path\n")

    def __len__(self):
        return self.size

    '''
    ->  each object has 2 dictionaries, each for a single clip with the following structure:
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
    # returns a list of objects with their labels - between 2 to 4 objects
    def __getitem__(self, index):
        pickle_idx = str(index).zfill(6) + '.pickle'
        file_path = os.path.join(self.dir_path, pickle_idx)
        try:
            mix_file = open(file_path, 'rb')
            pick = pickle.load(mix_file)
            mix_file.close()
        except OSError:
            print(file_path)
            self.log.write("-->> Error with file " + file_path)
            pick = None
            return pick

        X = pick
        pick_dict = {}
        # pick_dict['obj1'] = X['obj1']
        # pick_dict['obj2'] = X['obj2']
        # pick_dict['mix'] = X['mix']

        ids = [int(X['obj1']['id'])] + [int(X['obj2']['id'])]
        pick_dict['ids'] = np.vstack(ids)

        self_audios = [np.expand_dims(torch.FloatTensor(X['obj1']['audio']['stft'][0]), axis=0)]

        classes = [int(c[0]) for c in X['obj1']['images'][:]]
        if len(classes) == 1:
            classes += [16]
        else:
            self_audios += [np.expand_dims(torch.FloatTensor(X['obj1']['audio']['stft'][0]), axis=0)]

        self_audios += [np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][0]), axis=0)]
        
        classes += [int(c[0]) for c in X['obj2']['images'][:]]
        if len(classes) == 3:
            classes += [16]
        else:
            self_audios += [np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][0]), axis=0)]

        pick_dict['classes'] = np.vstack(classes)

        # ,
        #                np.expand_dims(torch.FloatTensor(X['obj1']['audio']['stft'][0]), axis=0),
        #                np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][0]), axis=0),
        #                np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][0]), axis=0)]  #array includes both videos data - 2 values
        
        pick_dict['audio_mags'] = np.vstack(self_audios)

        self_phases = [np.expand_dims(torch.FloatTensor(X['obj1']['audio']['stft'][1]), axis=0),
                            np.expand_dims(torch.FloatTensor(X['obj1']['audio']['stft'][1]), axis=0),
                            np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][1]), axis=0),
                            np.expand_dims(torch.FloatTensor(X['obj2']['audio']['stft'][1]), axis=0)]  #array includes both videos data - 2 values
        pick_dict['audio_phases'] = np.vstack(self_phases)


#Image.fromarray(c[1].astype(np.uint8), 'RGB')
        detected_objects = [self.transform(c[1][:,:,::-1] / 255).unsqueeze(0) for c in X['obj1']['images'][:]]
        # if len(detected_objects) == 1:
        #     detected_objects += [0 * detected_objects[0]]

        detected_objects += [self.transform(c[1][:,:,::-1] / 255).unsqueeze(0) for c in X['obj2']['images'][:]]
        # if len(detected_objects) == 3:
        #     detected_objects += [0 * detected_objects[2]]    #all detected objects in both video's'

        num_objs = len(detected_objects)




     #detected_objects /= 255
     
        # plt.figure()
        # im = detected_objects[0][0].T#.detach().cpu().numpy()
        # plt.imshow(im)# / (2550))
        # plt.show()
        # plt.savefig('./detect.png')

        pick_dict['detections'] = np.vstack(detected_objects)

        # for im in pick_dict['detections']:
        #     #im = im.reshape((224, 224, 3))
        #     im = im[1] / 255
        #     print(im)
        #     plt.imshow(im)
        #     plt.show()
        #     plt.savefig('./detection.png')
        #     break

        mixed_audio = []
        mix = X['mix'][0]
        mix = np.expand_dims(mix, axis=0)
        for n in range(num_objs):
            mixed_audio.append(torch.FloatTensor(mix).unsqueeze(0))
        
        pick_dict['mixed_audio'] = np.vstack(mixed_audio)

        mixed_phases = []
        mix_p = X['mix'][1]
        mix_p = np.expand_dims(mix_p, axis=0)
        for n in range(num_objs):
            mixed_phases.append(torch.FloatTensor(mix_p).unsqueeze(0))
        
        #mixed_audio = np.vstack(mixed_audio)
        #mixed_audio = mixed_audio + 1e-10  # in order to make sure we don't divide by 0
        #mixed_audio = mixed_audio
        pick_dict['mixed_phases'] = np.vstack(mixed_phases)

        pick_dict['rate'] = X['obj1']['audio']['wave'][1]

        return pick_dict


'''

separate for validation

find jpgs mean and std
apply trsfm in music dataset
normalize:  check for audio when to norm
            mix first and then normalize -1, 1 and then stft
move tensor part to getitem
pad with 0 or -1 and have 4 objects every time
define __len__ in dataset?
p
'''
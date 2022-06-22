"""
Important: we might have 2 classes with the same bbox due to the Detector results
"""

#import torch
#import torchvision
import sys
#import h5py
#import cv2
from PIL import Image
import wave
import librosa
import random
import numpy as np
import soundfile as sf
import os
import pickle
from datetime import datetime as dt

def sample_wav(wav, size=65535):
    # we expand the audio if its too short (with tile)
    if wav.shape[0] < size:
        n = int(size / wav.shape[0]) + 1
        wav = np.tile(wav, n)
    #start the sampling somewhere randomly
    start = random.randrange(0, wav.shape[0] - size + 1)
    #we get the audio values with the window size
    sample = wav[start:start+size]
    return sample


def create_spectrogram(wav, frame=1022, hop=256):
    trans = librosa.core.stft(wav, n_fft=frame, hop_length=hop, center=True)
    mag, phase = librosa.core.magphase(trans)
    #we treat it as array of arrays:
    mag = np.expand_dims(mag, axis=0)
    phase = np.expand_dims(np.angle(phase), axis=0)
    return mag, phase

def validChunk(path, parent):
    f1, f2, f3, f4, f5, f6, f7, f8 = False, False, False, False, False, False, False, False

    crop_name = ""

    for file in os.listdir(path):
        if file.endswith('.npy'):
            f1 = True
        if file == 'images':
            f2 = True
        if file == 'detection_results':
            f3 = True
        if file.startswith('wav'):
            f4 = True
        if file.startswith('cropped_'):
            f7 = True
            crop_name = file
    if f3:
        for file in os.listdir(os.path.join(path, 'detection_results')):
            if file == '.npy':
                f5 = True
    if f2:
        f6 = len(os.listdir(os.path.join(path, 'images'))) > 0

    if f7:
        crop = len(os.listdir(os.path.join(path, crop_name)))
        if parent == 'Duet':
            f8 = crop == 2
        elif parent == 'Solo':
            f8 = crop == 1

    res = f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8
    #print( "-->> " + path + " : " + str(res))

    return res

def overlap(b1, b2):
    pass

'''
->  For a given chunk path:
    Retrieves a dictionary for a single clip with the following structure:
    {
        'id'     :   video id number
        'audio'  :   { 'wave' : (wave, sr), 'stft' : (mags, phases) }
        'images' :   [(class_id1, image), (class_id2, image),...] -> just 1 or 2 cropped images
    }
'''
def pickItems(path, log, parent):
    if path is None:
        return None
    if not validChunk(path, parent):
        log.write("-> " + path + " is not a valid chunk")
        return None
    print("-->> " + path + " : " + str(True))

    obj_dict = {}
    wav_name = ""
    crop_name = ""

    for item in os.listdir(path):
        if item.lower().startswith("cropped_"):
            crop_name = item
        if item.lower().startswith("wav"):
            wav_name = item

    #a tuple of audio, sr
    audio, sr = librosa.load(os.path.join(path, wav_name), sr=11025)
    '''
    min_a = min(audio)
    max_a = max(audio)
    audio = 2 * (audio - min_a) / (max_a - min_a) - 1
    '''
    sample = sample_wav(audio)
    mags, phases = create_spectrogram(sample)
    obj_dict['audio'] = {'wave': (audio, sr), 'stft': (mags, phases)}

    obj_dict['images'] = []
    crop_path = os.path.join(path, crop_name)
    for bbox in os.listdir(crop_path):
        im = Image.open(os.path.join(crop_path, bbox)).resize((224, 224))
        pix = np.asarray(im).astype('float32')
        '''
        min_p = pix.min()
        max_p = pix.max()
        pix = (pix - min_p) / (max_p - min_p)
        '''
        obj_dict['images'] += [(bbox.split('.')[0], pix)]

    vid_id = crop_name.split('_')[-1]
    obj_dict['id'] = vid_id

    return obj_dict

#retrieves a random different clip from the video class
def pick_rand_clip(vid_class, vid_id, base_path):
    path = base_path
    if random.randrange(0, 2) == 0:
        path = os.path.join(path, 'Solo')
        parent = 'Solo'
    else:
        path = os.path.join(path, 'Duet')
        parent = 'Duet'

    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_cls = dirs[idx]
    while len(dirs) > 1 and (dir_cls == vid_class or dir_cls == '99'):
        dirs.remove(dir_cls)
        idx = random.randrange(0, len(dirs))
        dir_cls = dirs[idx]
    path = os.path.join(path, dir_cls)

    if dir_cls == '99':
        return None

    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_id = dirs[idx]
    while len(dirs) > 1 and dir_id == vid_id:
        dirs.remove(dir_id)
        idx = random.randrange(0, len(dirs))
        dir_id = dirs[idx]
    path = os.path.join(path, dir_id)

    # chunk select
    dirs = os.listdir(path)
    idx = random.randrange(0, len(dirs))
    dir_chunk = dirs[idx]
    while len(dirs) > 1 and not validChunk(os.path.join(path, dir_chunk), parent):
        dirs.remove(dir_chunk)
        idx = random.randrange(0, len(dirs))
        dir_chunk = dirs[idx]
        #print(os.path.join(path, dir_chunk))

    chunk_path = None
    if validChunk(os.path.join(path, dir_chunk), parent):
        chunk_path = os.path.join(path, dir_chunk)
    return chunk_path, parent

def iterate_files(dir, count, log, parent, source, target=r'/dsi/gannot-lab/datasets/Music/Batches'):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)

        if file == 'Duet':
            parent = 'Duet'
        if file == 'Solo':
            parent = 'Solo'

        #Erhu detection performed really worse and is not considered a genuine musical instrument detection
        if file == "99":
            continue

        if file.lower().startswith('chunk'):
            if not validChunk(file_path, parent):
                log.write("\n-->> " + file_path + " : not a vliad chunk\n")
                continue

            try:
                print(count)
                count[0] += 1

                obj1 = pickItems(file_path, log, parent)
                if obj1 is None:
                    log.write("\n-->> obj-1 : could not pick items for " + file_path + "\n")
                    continue

                random_clip_path = None
                #random_clip_path = file_path

                c = 0
                vid_id = file_path.split('/')[-2]
                vid_class = file_path.split('/')[-3]
                prob = 50

                while random_clip_path == None and c < prob:
                    random_clip_path, parent2 = pick_rand_clip(vid_class, vid_id, source)
                    c += 1
                obj2 = pickItems(random_clip_path, log, parent2)
                if obj2 is None:
                    log.write("-->> obj-2 : could not pick items for " + str(random_clip_path) + "\n")
                    continue
                mix_stft = (obj1['audio']['wave'][0] + obj2['audio']['wave'][0]) / 2

                #mix_stft = librosa.util.normalize(mix_stft)

                min_a = min(mix_stft)
                max_a = max(mix_stft)
                mix_stft = 2 * (mix_stft - min_a) / (max_a - min_a) - 1

                sample = sample_wav(mix_stft)
                mix_mags, mix_phases = create_spectrogram(sample)

                #assume target exists
                t_path = os.path.join(target, str(count[1]).zfill(6) + ".pickle")
                count[1] += 1
                obj_dict = {
                    "obj1": obj1,
                    "obj2": obj2,
                    "mix": (mix_mags, mix_phases)
                }

                with open(t_path, 'wb') as handle:
                    pickle.dump(obj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    log.write("\n---------------------------------------------------------------------------\n[" +
                              str(count[1]).zfill(6) + "] -->> " + file_path +
                              "\n\n" +
                              "         -->> " + random_clip_path +
                              "\n---------------------------------------------------------------------------\n")
            except Exception as exc:
                log.write("error with file " + file_path + " with : " + str(exc) + "\n")

        elif os.path.isdir(file_path):
            iterate_files(file_path, count, log, parent, source, target)


#pick for each video random other video and combine them together and normalize the audio

if __name__ == "__main__":
    #print("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/GeneratorErrorsLog.txt", "x")
    except:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/GeneratorErrorsLog.txt", "w")

    log.write("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")
    log.write("\nGenerator Errors : \n")

    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    count = [0, 0]
    source = root_dir           #'/dsi/gannot-lab/datasets/Music/Try/'
    iterate_files(root_dir, count, log, '', source)
    print("\n---------------SECOND ROUND------------------\n")
    log.write("\n---------------SECOND ROUND------------------\n")
    iterate_files(root_dir, count, log, '', source)


'''
    #f = h5py.File(r'C:/Users/user/Desktop/try.h5', 'a')
    #im = cv2.imread(r'C:/Users/user/Desktop/0.jpg')
    #im = Image.open(r'C:/Users/user/Desktop/0.jpg')
    #im = im.crop((0, 0, 300, 300))

    #im.show()

    #im = im.resize((224, 224))
    #print(im.size)
    #im.show()

    #im = im.resize((224, 224))
    #print(im.size)
    #im.show()

    wav, rate = librosa.load('C:/Users/user/Desktop/wav_12.wav', sr=11025)
    wav2, rate2 = librosa.load('C:/Users/user/Desktop/wav_4.wav', sr=11025)
    wav3 = (librosa.util.normalize(wav) + librosa.util.normalize(wav2))
    wav4 = librosa.util.normalize(wav3)
    print(min(wav4))
    wav5 = librosa.util.normalize(wav2)
    print(min(wav5))
    sf.write('C:/Users/user/Desktop/wav_13.wav', wav4, rate)
    sf.write('C:/Users/user/Desktop/wav_5.wav', wav5, rate)

    a = 1
    if a == 1:
        exit(0)

    #wav = Wave('C:/Users/user/Desktop/wav_1.wav')
    #wav.start()
    #wav.overlay(wav2)
    wav3 = (wav + wav2) / 2

#    sf.write('C:/Users/user/Desktop/wav_11.wav', wav3, rate)
    #audio.write_audiofile(filename=chunkPath + '/wav_' + str(index) + '.wav', codec='pcm_s32le')
    #wav = librosa.stft(wav)
    print("rate = " + str(rate))
    print(wav.shape)
    size = 65535
    samp = sample_wav(wav, size)

    print("Length = ", end='')
    print(len(samp))
    frame = 1022
    hop = 256
    mag, phase = create_spectrogram(samp, frame, hop)
    print(mag.shape)
    print(phase.shape)




    #print(im.shape)
    #f.create_dataset(name="image", data=im)
    #f.create_dataset(name="audio_mag", data=mag)

    print(list(f.keys()))
    print(f['audio_mag'][0])
    #print(f['image'])'''
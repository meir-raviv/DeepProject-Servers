#pip install opencv-python
#pip install --trusted-host pypi.python.org moviepy
#pip install imageio-ffmpeg
#pip install ffmpeg-python
#pip install pygame


import os
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from numpy import timedelta64
#import pygame
#from moviepy.video.VideoClip import VideoClip
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip
from MusicDataset import MusicDataset
import time


class DataLoader():
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode
    
    def splitData(self):
        train, test = [], []
        return train, test

    def sampleFrames(self, clip, fps):
        frames = []
        t = 0
        while t < clip.duration:
            frames += [clip.get_frame(t)]
            t += 1 / fps
        return frames

    def separateSubClip(self, clip, filePath, offset, duration, fps, index):
        
        #clip = VideoFileClip(filePath)
        clip = clip.subclip(offset, offset + duration)
        #clip.preview()
        audio = clip.audio
        #wav = os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format("wav", filePath))
        audio.write_audiofile(filename='Data2/wav.wav', codec='pcm_s32le')
        time.sleep(0.3)
        try:
            audio = librosa.load('Data2/wav.wav', offset=0, duration=duration, sr=44100)
        except:
            audio = None
            print("false")
        #input("enter:")
        # audio.stream()
        #audio1 = librosa.load(audio)
        # #librosa.stream(audio1)
        # plt.figure(figsize=(14, 5))
        # librosa.display.waveplot(audio)
        #audio.preview()
        frames = self.sampleFrames(clip, fps)
        # for i in range(0, len(frames)):
        #     plt.imshow(frames[i])
        #     plt.show()
        dict = {'chunk' : index, 'audio' : audio, 'frames' : frames}

        return dict

    def separateClip(self, filePath, fps):
        
        clip = VideoFileClip(filePath)
        duration = 10
        offset = 0
        dict = []
        index = 0
        while offset + duration < clip.duration:
            dict += [self.separateSubClip(clip, filePath, offset, duration, fps, index)]
            offset += 10
            index += 1

        return dict

    def loadDataFromFile(self, path, fps):
        filesList = os.listdir(path)
        data = []
        i = 0
        for file in filesList:
            if file[len(file) - 3:len(file)] == 'mp4':
                val = self.separateClip(path + '/' + file, fps)
                #audio.preview()
                dict = {'file index' : i, 'value' : val}
                i += 1
                data += [dict]

        return data

if __name__ != 'main':
    fps = 8

    print("k")
    
    DL = DataLoader("", "test")
    data = DL.loadDataFromFile('Data2', fps)




    # for i in range(len(data[0]['value'])):
    #     x = data[0]['value'][i]['audio']
        #pass
        # #print(x)
        # y = x[0]
        # fig, ax = plt.subplots()#nrows=1, ncols=1, sharex=True)
        # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        # img = librosa.display.specshow(D, sr=44100)
        # fig.colorbar(img, ax=ax, format="%+2.f dB")
        # fig.show()
        #plt.imshow(img)

    #cap = cv2.VideoCapture('Data2/jam.wav')
    # sr = 44100
    # y, sr = librosa.load('Data2/wav.wav', offset=0, duration=10, sr=44100)
    # D = librosa.stft(y)
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(S_db, ax=ax)
    # fig.colorbar(img, ax=ax)
    # fig.show()
    # input()
    #print(len(y))
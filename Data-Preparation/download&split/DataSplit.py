#pip install opencv-python
#pip install --trusted-host pypi.python.org moviepy
#pip install imageio-ffmpeg
#pip install ffmpeg-python
#pip install pygame


import os
#from matplotlib import pyplot as plt
import numpy as np
#import librosa
#import librosa.display
#from numpy import timedelta64
#import pygame
#from moviepy.video.VideoClip import VideoClip
from moviepy.editor import VideoFileClip
#from moviepy.editor import AudioFileClip
#from MusicDataset import MusicDataset
#import time


class DataSpliter():
    def __init__(self, fps, duration):
        self.duration = duration
        self.fps = fps
    
    def sampleFrames(self, clip):
        frames = []
        t = 0
        while t < clip.duration:
            frames += [clip.get_frame(t)]
            t += 1 / self.fps

        return frames

    def separateSubClip(self, clip, trg_path, offset, index):
        chunkPath = trg_path + "/chunk_" + str(index)
        try:
            os.mkdir(chunkPath)
        except:
            print("error creating chunk directory" + chunkPath)

        #clip = VideoFileClip(filePath)
        clip = clip.subclip(offset, offset + self.duration)
        #clip.preview()
        audio = clip.audio
        #wav = os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format("wav", filePath))
        
        audio.write_audiofile(filename=chunkPath + '/wav_' + str(index) + '.wav', codec='pcm_s32le')
        
        # time.sleep(0.001)
        # try:
        #     audio = librosa.load(chunkPath + '/wav_' + str(index) + '.wav', offset=0, duration=self.duration, sr=44100)
        # except:
        #     audio = None
        #     print("false")
        # #input("enter:")
        # audio.stream()
        #audio1 = librosa.load(audio)
        # #librosa.stream(audio1)
        # plt.figure(figsize=(14, 5))
        # librosa.display.waveplot(audio)
        #audio.preview()
        frames = self.sampleFrames(clip)
        chunk_file = chunkPath + '/frames_' + str(index)
        # try:
        #     file = open(chunk_file, "x")
        # except:
        #     print("Exists")
        #     try:    
        #         file = open(chunk_file, "a")
        #     except:
        #         print("bad error")

        
#        np.save(file, frames)

        np.save(chunk_file, frames)

        # for i in range(0, len(frames)):
        #     plt.imshow(frames[i])
        #     plt.show()
        # dict = {'chunk' : index, 'audio' : audio, 'frames' : frames}

        # return dict

    def separateClip(self, src_path, trg_path):
        
        clip = VideoFileClip(src_path)
        #duration = 10
        offset = 0
        #dict = []
        index = 0
        while offset + self.duration < clip.duration:
            # print("input 3:")
            # x = input()
            self.separateSubClip(clip, trg_path, offset, index)
            offset += 10
            index += 1

        #return dict

    def separateDataOfFile(self, file, src_path, trg_path):
        #filesList = os.listdir(path)
        #data = []
        #i = 0
        #for file in filesList:
        if file[len(file) - 3:len(file)] == 'mp4':
            # print("input 2:")
            # x = input()
            self.separateClip(src_path, trg_path)
            #x = input()
            #audio.preview()
                #dict = {'file index' : i, 'value' : val}
                #i += 1
                #data += [dict]

#        return data

    def separateAllDataInDirectory(self, src_path, trg_path):
        filesList = os.listdir(src_path)
        for file in filesList:
            # print("input 1:")
            # x = input()
            filePath = os.path.join(src_path, file)
            name = file[0 : len(file) - 4]
            if os.path.isdir(filePath):
                if file != "wrong" and file != "Let Me Love You - Saxophone & Guitar Cover - BriansThing & Anna Sentina ??.mp4":
                    try:
                        os.mkdir(trg_path + "/" + file)
                    except:
                        print("Directory already exists")
                    self.separateAllDataInDirectory(filePath, trg_path + "/" + file)
            else:
                try:
                    if name != "Let Me Love You - Saxophone & Guitar Cover - BriansThing & Anna Sentina ??":
                        try:
                            os.mkdir(trg_path + "/" + name)
                        except:
                            print("Directory already exists")
                        #os.rename(filePath, trg_path + "/" + name + file)
                        self.separateDataOfFile(file, src_path + "/" + file, trg_path + "/" + name)
                except:
                    print("error with video" + file)
        



fps = 8
duration = 10

Source_Path = "/dsi/gannot-lab/datasets/Music"
Target_Path = Source_Path + "/MUSIC_arme/Data"

DL = DataSpliter(fps, duration)
data = DL.separateAllDataInDirectory(Source_Path + "/Data", Target_Path)




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

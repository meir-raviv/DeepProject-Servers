import json
import DataDownloader as datd
import DataLoader as datl
import numpy as np
import torch
from torch.utils.data import DataLoader
from NetModel import CoSeparationNet
from ResNet101 import ResNet101
from AudioVisualSeparator import AudioVisualSeparator


'''Download Data from YouTube'''
# path = "Data"
# jsonObj = json.load(open('MUSIC.json'))
# errorLog = datd.downloadDataFromJSON(path, jsonObj)

'''Data Split'''

offset = 10
duration = 1
fps = 8


filePath = 'Data2'
data = datl.loadDataFromFile(filePath, offset, duration, fps)

'''Load Data'''


'''Arrange cuda'''
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

'''Build Model'''
model = CoSeparationNet(ResNet101(), )
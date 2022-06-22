import torch
import torchvision
import torchvision.transforms as T
import sys
import os
import subprocess
from datetime import datetime
from datetime import datetime as dt
import numpy as np
from PIL import Image

def iterate_files(dir, count, pic_tensor):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            print(count)
            for item in os.listdir(file_path):
                if item.lower().startswith("cropped_"):
                    count[0] += 1
                    pics_path = os.path.join(file_path, item)
                    for pic in os.listdir(pics_path):
                        pic_path = os.path.join(pics_path, pic)
                        if pic.endswith(('.jpg')):
                            count[1] += 1
                            im = Image.open(pic_path)
                            tim = T.ToTensor()(im)
                            pic_tensor += tim


        elif os.path.isdir(file_path):
            iterate_files(file_path, count, pic_tensor)


if __name__ == "__main__":
    # argument 1 is the root directory of the data

    path = r"/dsi/gannot-lab/datasets/Music/MUSIC_arme/Data/Duet/11_15/000000/chunk_0/cropped_000000/8.jpg"
    im1 = Image.open(path)
    tim1 = T.ToTensor()(im1)

    root_dir = r"/dsi/gannot-lab/datasets/Music/MUSIC_arme/Data/"   #sys.argv[1]
    count = [0, 0]
    pic_tensor = tim1
    pic_tensor -= tim1
    iterate_files(root_dir, count, pic_tensor)
    a = pic_tensor / count[1]
    m = a.mean((1, 2))
    s = a.std((1, 2))
    print(m)
    print(s)

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/Mean&Std.txt", "x")
    except OSError:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/Mean&Std.txt", "w")

    log.write("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")
    log.write("\nMean & Std : \n")
    log.write(str(m) + "\n")
    log.write(str(s) + "\n")
    log.write(str(count[1]) + " pictures\n")



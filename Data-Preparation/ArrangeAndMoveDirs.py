import torch
import torchvision
import sys
import os
import numpy as np
import shutil


def iterate_files(dir, dest):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            for item in os.listdir(dir):
                if item.lower().endswith('.wav'):
                    wav_file_path = os.path.join(dir, item)
                    shutil.copy(wav_file_path, file_path)

                    #copy dir to data folder
                    destDir = os.path.join(dest, file)
                    os.mkdir(destDir)
                    for subs in os.listdir(file_path):
                        sub_path = os.path.join(file_path, subs)
                        os.copy(sub_path, destDir)

        elif os.path.isdir(file_path):
            iterate_files(file_path, dest)


if __name__ == "main":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    dest = sys.argv[2]
    iterate_files(root_dir, dest)


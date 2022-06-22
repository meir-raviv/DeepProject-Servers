import torch
import torchvision
import sys
import os
import subprocess


def iterate_files(dir, count):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            print(count)
            count[0] += 1
            flag = False
            for item in os.listdir(file_path):
                if item.lower().startswith('images'):
                    flag = True
            if not flag:
                print(file_path)
        elif os.path.isdir(file_path):
            iterate_files(file_path, count)

if __name__ == "__main__":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    x = [0]
    iterate_files(root_dir, x)
    print("Done")


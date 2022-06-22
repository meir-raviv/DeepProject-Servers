import torch
import torchvision
import sys
import os
import shutil


def iterate_files(dir, count):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            print(count)
            count[0] += 1
            for item in os.listdir(file_path):
                if item.lower().startswith('images'):
                    for img in os.listdir(os.path.join(file_path, item)):
                        if img.startswith("39"):
                            try:
                                os.mkdir(os.path.join(file_path, "image"))
                            except OSError:
                                pass
                            shutil.copyfile(os.path.join(file_path, item, img), os.path.join(file_path, "image", img))
        elif os.path.isdir(file_path):
            iterate_files(file_path, count)

if __name__ == "__main__":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    x = [0]
    iterate_files(root_dir, x)
    print("Done")


import torch
import torchvision
import sys
import os
import subprocess

classes_dict = {'Banjo':1, 'Cello':2, 'Drum':3, 'Guitar':4,
                       'Harp':5, 'Harmonica':6, 'Oboe':7, 'clarinet':7, 'Piano':8, 'xylophone':8, 'Saxophone':9,
                       'Trombone':10, 'Trumpet':11, 'Violin':12, 'Flute':13,
                       'Accordion':14, 'Horn':15, 'tuba':15, 'erhu':99}

def iterate_files(root):
    count = 0
    for dir1 in sorted(os.listdir(root)):
        for dir2 in sorted(os.listdir(os.path.join(root, dir1))):
            for dir3 in sorted(os.listdir(os.path.join(root, dir1, dir2))):
                file_path = os.path.join(root, dir1, dir2, dir3)
                new_name = os.path.join(root, dir1, dir2, str(count).zfill(6))
                count += 1
                print("rename " + file_path + " to " + new_name)
                os.rename(file_path, new_name)


if __name__ == "__main__":
    # argument 1 is the root directory of the data
    root_dir = sys.argv[1]
    print(root_dir)
    iterate_files(root_dir)
    #print(classes_dict)
    #print("2_4".split("_"))

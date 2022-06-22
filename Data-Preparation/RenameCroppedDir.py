import torch
import torchvision
import sys
import os
import subprocess
from datetime import datetime
from datetime import datetime as dt

def iterate_files(dir, count, log):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            #print(count)
            #count[0] += 1
            flag = False
            for item in os.listdir(file_path):
                if item.lower().startswith("cropped_"):
                    count[0] += 1
                    flag = True
                    new_name = os.path.join(file_path, "old-" + item)
                    old_name = os.path.join(file_path, item)
                    log.write("\n[" + str(count[0]) + " ] -->> rename " + old_name + " to " + new_name + "\n")
                    print("[" + str(count[0]) + " ] -->> rename " + old_name + " to " + new_name)
                    os.rename(old_name, new_name)
            if not flag:
                #print(file_path)
                count[1] += 1
        elif os.path.isdir(file_path):
            iterate_files(file_path, count, log)


if __name__ == "__main__":
    # argument 1 is the root directory of the data

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/RenameCroppedErrorsLog.txt", "x")
    except OSError:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/RenameCroppedErrorsLog.txt", "w")

    log.write("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")
    log.write("\nRename Cropped Errors : \n")

    root_dir = sys.argv[1]
    count = [0, 0]
    iterate_files(root_dir, count, log)

    # now = datetime.now()
    # for i in range(1000000):
    #     pass
    #print(datetime.now() - now)
    print("Done, number of chunks that have cropped_ = " + str(count[0]))
    print("      number of chunks that do not have cropped_ = " + str(count[1]))


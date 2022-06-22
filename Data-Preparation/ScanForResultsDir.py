import torch
import torchvision
import sys
import os
from datetime import datetime as dt
import subprocess
from datetime import datetime

def iterate_files(dir, count, log):

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if file.lower().startswith('chunk'):
            print(count)
            #count[0] += 1
            flag = False
            for item in os.listdir(file_path):
                if item == "detection_results":
                    count[1] += 1
                    flag = True
                    s = len(os.listdir(os.path.join(file_path, item)))
                    log.write("[" + str(count[1]) + "] --->>> " + str(s) + " detections in " + os.path.join(file_path, item) + "\n")
                    if s == 1:
                        count[2] += 1
            if not flag:
                #print(file_path)
                count[0] += 1
        elif os.path.isdir(file_path):
            iterate_files(file_path, count, log)


if __name__ == "__main__":
    # argument 1 is the root directory of the data

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/ScanDetectionsErrorsLog.txt", "x")
    except OSError:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/ScanDetectionsErrorsLog.txt", "w")

    log.write("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")
    log.write("\nScan Detections Errors : \n")

    root_dir = sys.argv[1]
    count = [0, 0, 0]
    iterate_files(root_dir, count, log)

    # now = datetime.now()
    # for i in range(1000000):
    #     pass
    #print(datetime.now() - now)
    print("Done, number of chunks that have detection_dir = " + str(count[1]))
    print("      number of chunks that have 1 file in detection_dir = " + str(count[2]))


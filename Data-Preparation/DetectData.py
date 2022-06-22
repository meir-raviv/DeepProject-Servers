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
            print(count)
            count[0] += 1
            flag = False
            s = 0
            for item in os.listdir(file_path):
                if item == "detection_results":
                    flag = True
                    s = len(os.listdir(os.path.join(file_path, item)))
                    if s != 1:
                        log.write("-> " + str(s) + " detections in " + os.path.join(file_path, item) + "\n")
            if not flag or s == 0:
                for item in os.listdir(file_path):
                    if item == "images":
                        images_path = os.path.join(file_path, item + '/')
                        print(images_path)
                        p = subprocess.call(["python",
                                              r"/home/dsi/ravivme/co-separation/getDetectionResults.py",
                                              "--cfg",
                                              r"/home/dsi/ravivme/fasterRCNN/faster-rcnn.pytorch/cfgs/res101_ls.yml",
                                              "--load_dir",
                                              r"/home/dsi/ravivme/fasterRCNN/faster-rcnn.pytorch/data/pretrained_model",
                                              "--net",
                                              "res101",
                                              "--checksession",
                                              "1",
                                              "--checkepoch",
                                              "1",
                                              "--checkpoint",
                                              "1",
                                              "--image_dir",
                                              images_path])

        elif os.path.isdir(file_path):
            iterate_files(file_path, count, log)


if __name__ == "__main__":
    # argument 1 is the root directory of the data

    try:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/DetectionsErrorsLog.txt", "x")
    except:
        log = open(r"/dsi/gannot-lab/datasets/Music/Logs/DetectionsErrorsLog.txt", "w")

    log.flush()
    log.write("\nTime Stamp : " + str(dt.date(dt.now())) + " , " + str(dt.now().strftime("%H:%M:%S")) + "\n")
    log.write("\nDetections Errors : \n")

    root_dir = sys.argv[1]
    iterate_files(root_dir, [0], log)
    # now = datetime.now()
    # for i in range(1000000):
    #     pass
    #print(datetime.now() - now)
    print("Done")


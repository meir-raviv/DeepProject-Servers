import numpy as np
import sys
import cv2
import os
import subprocess

#path = r"C:\Users\user\Desktop\frames"
path = sys.argv[1]
#fileName = "frames_4.npy"
fileName = sys.argv[2]
print(os.path.join(path, fileName))
arr = np.load(os.path.join(path, fileName))
# print(len(arr))
fullPath = os.path.join(path, "images")
try:
    os.mkdir(fullPath)
except OSError:
    pass
#for i, img in enumerate(arr):
    #print(i)

cv2.imwrite(os.path.join(fullPath, str(0) + ".jpg"), arr[0])
cv2.imwrite(os.path.join(fullPath, str(39) + ".jpg"), arr[39])
cv2.imwrite(os.path.join(fullPath, str(79) + ".jpg"), arr[-1])

exit(0)
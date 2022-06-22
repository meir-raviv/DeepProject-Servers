# import necessary libraries
import pandas as pd
import numpy as np

# create a dummy array
arr = np.load(r"C:\Users\user\Desktop\12.npy")
print(arr[0].shape == (7,))
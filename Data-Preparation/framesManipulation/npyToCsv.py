# import necessary libraries
import pandas as pd
import numpy as np

# create a dummy array
arr = np.load(r"C:\Users\user\Desktop\hi.npy")

# display the array
#print(arr)

# convert array into dataframe
DF = pd.DataFrame(arr)

# save the dataframe as a csv file
DF.to_csv(r"C:\Users\user\Desktop\detHi.csv")
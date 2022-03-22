
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os


array = np.array([[[1, 2,3], [4, 5,6], [7,8,9]],[[1, 2,3], [4, 5,6], [7,8,9]],[[1, 2,3], [4, 5,6], [7,8,9]]])
print("array" + str(array))
print(array.shape)
array_padded = np.pad(array, [(1,1), (1,1),(1,1)], mode = 'constant', constant_values = [(-1024,-1024),(-1024,-1024), (-1024,-1024)])
array_padded = array_padded.astype(np.int0)
print(array_padded)
print(array_padded.shape)

lst = [1,2,3]
print(len(lst))

import numpy as np
array = np.zeros([2,2], dtype = bool)

print(array)

test = [[True, False],
 [False ,False]]
array = array + test
print(array)
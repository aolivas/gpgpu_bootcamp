#!/usr/bin/env python

import numpy as np
from numba import cuda

@cuda.jit
def sum_array(result, data):
    idx = cuda.threadIdx.x
    for value in data[idx]:
        result[idx] += value
    for value in data[idx + cuda.blockDim.x]:
        result[idx] += value

n_arrays = 1024
array_size = 10000
np.random.seed(42)
data = np.random.random((n_arrays, array_size))                 
grid = (1,1,1)
block = (n_arrays/2,1,1)
result = np.zeros(block[0])
sum_array[grid, block](result, data)
print(sum(result))
print(np.sum(data))








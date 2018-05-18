#!/usr/bin/env python

import numpy as np
from numba import cuda

N = 8
@cuda.jit
def sum_array(result, data):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    s = 0
    for i in range(8):
        for j in range(8):
            s += data[N * x + i, N * y + j]
    result[x,y] += s
        
n_arrays = 1024
array_size = 1024
np.random.seed(42)
data = np.random.random((n_arrays, array_size))                 
grid = (8,8,1)
block = (16,16,1)
result = np.zeros((1024,1024)) # absolute thread dimensions
sum_array[grid, block](result, data)
print(np.sum(result))
print(np.sum(data))








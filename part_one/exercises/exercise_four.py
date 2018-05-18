#!/usr/bin/env python

import numpy as np
from numba import cuda

N = 8
@cuda.jit
def sum_array(result, data):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    s = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                s += data[N * x + i, N * y + j, N * z + k]
    result[x,y,z] += s
        
np.random.seed(42)
data = np.random.random((256, 256, 256))                 
grid = (4,4,4)
block = (8,8,8)
result = np.zeros((32, 32, 32)) # absolute thread dimensions
sum_array[grid, block](result, data)
print(np.sum(result))
print(np.sum(data))








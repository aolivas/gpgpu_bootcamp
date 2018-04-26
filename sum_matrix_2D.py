#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from numba import cuda

@cuda.jit
def sum_array(result, a, b):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    result[i][j] = a[i][j] + b[i][j]

a = np.array([[1,0],
              [0,1]])
b = np.array([[0,1],
              [1,0]])
r = np.array([[0,0],
              [0,0]])

sum_array[1,(2,2)](r, a, b)
print(r)

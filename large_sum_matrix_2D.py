#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from numba import cuda

@cuda.jit
def sum_array(result, a, b):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    result[i][j] = a[i][j] + b[i][j]

N = 8192
a = np.random.random((N,N))
b = np.random.random((N,N))
r = np.zeros((N,N))
sum_array[1,(2,2)](r, a, b)
print(r)
#sum_array[1,(8192,8192)](r, a, b)
#sum_array[(128,128),(64,64)](r, a, b)
sum_array[(256,256),(32,32)](r, a, b)
print("...done.")
print(r)
sum_array[(512,512),(16,16)](r, a, b)
print("...done.")
print(r)
sum_array[(1024,1024),(8,8)](r, a, b)
print("...done.")
print(r)
sum_array[(512,1024),(16,8)](r, a, b)
print("...done.")
print(r)
sum_array[(1024,512),(8,16)](r, a, b)
print("...done.")
print(r)



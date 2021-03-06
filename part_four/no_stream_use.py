#!/usr/bin/env python

from numba import cuda
import numpy as np

@cuda.jit
def sum_array(a, b, c):
    i,j = cuda.grid(2) 
    c[i][j] = a[i][j] + b[i][j]

N = 4096
block = (32,32)
grid = (128, 128)
n_launches = 2
A = np.random.random((n_launches, N, N))
B = np.random.random((n_launches, N, N))
C = np.zeros((n_launches, N, N))

cuda.profile_start()
for a,b,c in zip(A,B,C):
    sum_array[grid, block](a,b,c) 
cuda.profile_stop()

print(C)




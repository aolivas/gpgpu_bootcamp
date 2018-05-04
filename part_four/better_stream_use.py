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
streams = [cuda.stream() for _ in range(n_launches)]

cuda.profile_start()
for idx,(a,b,c, stream) in enumerate(zip(A,B,C,streams)):
    da = cuda.to_device(a, stream)
    db = cuda.to_device(b, stream)
    dc = cuda.to_device(c, stream)
    sum_array[grid, block, stream](da,db,dc)
    C[idx] = dc.copy_to_host()
cuda.profile_stop()

print(C)


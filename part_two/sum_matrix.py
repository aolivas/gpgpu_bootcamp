#!/usr/bin/env python

from __future__ import print_function

import time, sys
import numpy as np

from numba import cuda

@cuda.jit
def sum_array(result, a, b):
    i,j = cuda.grid(2) 
    result[i][j] = a[i][j] + b[i][j]

N = 8192
a = np.random.random((N,N))
b = np.random.random((N,N))
r = np.zeros((N,N))
sum_array[1,(2,2)](r, a, b)

M = int(sys.argv[1])
K = int(sys.argv[2])
block = (M,K)
grid = ((N + M - 1)/M, (N + K -1)/K)
cuda.profile_start()
start_time = time.time()
sum_array[grid, block](r, a, b)
end_time = time.time()
cuda.profile_stop()
print("sum_array[%s, %s] elapsed time = %.2f ms" %
      (str(grid), str(block), (1e3*(end_time - start_time))))



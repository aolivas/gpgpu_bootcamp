#!/usr/bin/env python

from numba import cuda
import numpy as np

n_arrays = 4096
array_size = 10000
np.random.seed(42)
A = np.random.random((array_size, n_arrays))
B = np.random.random((array_size, n_arrays))
C = np.zeros((array_size, n_arrays))

@cuda.jit
def sum_array(a, b, c):
    i,j = cuda.grid(2) 
    c[i][j] = a[i][j] + b[i][j]

cuda.profile_start()
stream = cuda.stream()
ptr = cuda.to_device(data, stream)
#ptr.copy_to_device(data, stream=stream)
#noop[1,1,stream](data)
cuda.profile_stop()


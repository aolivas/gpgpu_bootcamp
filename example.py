#!/usr/bin/env python

from __future__ import print_function

from numba import cuda
import numpy as np

@cuda.jit
def max_example(result, values):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    print(tid)
    cuda.atomic.max(result, 0, values[i])

arr = np.random.rand(16384)
result = np.zeros(1, dtype=np.float64)

max_example[32,32](result, arr)
print(result[0])
print(max(arr))

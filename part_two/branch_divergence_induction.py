#!/usr/bin/env python

from __future__ import print_function

import time
from math import cos, sin
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def sum_cos(array):
    result = 0.
    for v in array:
        result += cos(v)
    return result

@cuda.jit(device=True)
def sum_sin(array):
    result = 0.
    for v in array:
        result += sin(v)
    return result

@cuda.jit
def sum_array(result, data):
    idx = cuda.threadIdx.x
    array = data[idx]
    branch = bool(idx % 2)
    #branch = bool((idx / 32) % 2)
    #branch = bool(idx % 31 == 0)
    if branch:
        r = sum_cos(array)
    else:
        r = sum_sin(array)
    result[idx] = r
        
n_arrays = 1024 
array_size = 10000
np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

result = np.zeros(n_arrays)
start_time = time.time()
sum_array[1, n_arrays](result, data)
end_time = time.time()
print(sum(result))
print("elapsed time = %.2f ms" % (1e3*(end_time - start_time)))










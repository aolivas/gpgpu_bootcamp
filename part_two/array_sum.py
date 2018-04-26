#!/usr/bin/env python

from __future__ import print_function

import time
from math import cos
import numpy as np
from numba import cuda

@cuda.jit
def sum_coaine(result, data):
    idx = cuda.threadIdx.x 
    array = data[idx]
    for value in array:
        result[idx] += cos(value)a

n_arrays = 1024
array_size = 10000
np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

result = np.zeros(n_arrays)
start_time = time.time()
sum_array[100, n_arrays](result, data)
end_time = time.time()
print(sum(result))
print("elapsed time = %.2f ms" % (1e3*(end_time - start_time)))










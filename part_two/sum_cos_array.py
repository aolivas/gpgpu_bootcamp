#!/usr/bin/env python

from __future__ import print_function

import time
from math import cos
import numpy as np
from numba import cuda

@cuda.jit
def cosine(result, data):
    idx = cuda.threadIdx.x 
    result[idx] = cos(data[idx])
    #result[idx+1] = cos(data[idx+1])
    #result[idx+2] = cos(data[idx+2])
    #result[idx+3] = cos(data[idx+3])

array_size = 1024
np.random.seed(42)
data = np.random.random(array_size)
result = np.zeros(array_size)

start_time = time.time()
cosine[1, array_size](result, data)
end_time = time.time()
print(sum(result))
print("elapsed time = %.2f ms" % (1e3*(end_time - start_time)))











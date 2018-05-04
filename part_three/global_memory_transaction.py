#!/usr/bin/env python

import sys
import numpy as np
from numba import cuda

offset = int(sys.argv[1])
array_size = 10000

np.random.seed(42)
data = np.random.random(array_size)
result = np.zeros(array_size)

@cuda.jit
def kernel(result, array, offset):
    idx = cuda.threadIdx.x + offset 
    result[idx + offset] = 10. * array[idx] 
    
cuda.profile_start()
kernel[1,1024](result, data, offset)
cuda.profile_stop()

    


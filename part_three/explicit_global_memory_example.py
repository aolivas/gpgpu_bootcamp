#!/usr/bin/env python

import numpy as np
from numba import cuda

n_arrays = 4096
array_size = 10000
size = 8 * array_size * n_arrays
print("Need at least %d MB for this array" % (float(size)/1e6))

np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

@cuda.jit
def kernel(array):
    idx = cuda.threadIdx.x

cuda.profile_start()
device_array = cuda.to_device(data)
kernel[1,1](device_array)
#result_array = device_array.copy_to_host()
cuda.profile_stop()


    

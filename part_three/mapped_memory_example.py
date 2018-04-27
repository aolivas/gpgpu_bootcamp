#!/usr/bin/env python

import numpy as np
from numba import cuda

n_arrays = 4096
array_size = 10000
size = 8 * array_size * n_arrays
print("Need at least %d MB for this array" % (float(size)/1e6))

np.random.seed(42)
data = cuda.mapped_array((4096, 10000), dtype=np.float)
for i in range(n_arrays):
    data[i] = np.random.random(array_size)
    
@cuda.jit
def kernel(array):
    idx = cuda.threadIdx.x

cuda.profile_start()
device_array = cuda.to_device(data)
kernel[1,1](device_array)
cuda.profile_stop()


    

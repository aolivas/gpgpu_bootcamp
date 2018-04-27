#!/usr/bin/env python

import numpy as np
from numba import cuda

n_arrays = 4096
array_size = 10000
np.random.seed(42)
# fills array with float64
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])
size = 8 * array_size * n_arrays
print("Need at least %d MB for this array" % (float(size)/1e6))

global_device_array = cuda.to_device(data)
pinned_host_array = cuda.pinned_array((n_arrays, array_size))

@cuda.jit
def kernel():

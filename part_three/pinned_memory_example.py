#!/usr/bin/env python

import numpy as np
from numba import cuda

n_arrays = 4096
array_size = 10000
np.random.seed(42)
data = cuda.pinned_array((n_arrays, array_size))
for i in range(n_arrays):
    data[i] = np.random.random(array_size)
    
cuda.profile_start()
device_array = cuda.to_device(data)
cuda.profile_stop()


    

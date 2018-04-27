#!/usr/bin/env python
import time
from numba import cuda
import numpy as np

n_arrays = 4096
array_size = 10000
np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

@cuda.jit
def noop(array):
    idx = cuda.threadIdx.x

cuda.profile_start()
start_time = time.time()
stream = cuda.stream()
ptr = cuda.to_device(data, stream)
#ptr.copy_to_device(data, stream=stream)
#noop[1,1,stream](data)
end_time = time.time()
cuda.profile_stop()
print("elapsed time = %.2f ms" % (1e3*(end_time - start_time)))

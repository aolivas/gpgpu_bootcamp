#!/usr/bin/env python

import numpy as np
from numba import cuda

@cuda.jit
def sum_array(result, data):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    array = data[idx]
    for value in array:
        result[idx] += value

n_arrays = 1024
array_size = 10000
np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

result1 = np.zeros(n_arrays)
result2 = np.zeros(n_arrays)
result4 = np.zeros(n_arrays)
sum_array[1, n_arrays](result1, data)
sum_array[2, n_arrays/2](result2, data)
sum_array[4, n_arrays/4](result4, data)
print(sum(result1))
print(sum(result2))
print(sum(result4))

#import cProfile
#pr = cProfile.Profile()
#pr.enable()
#result = np.sum(data)
#pr.disable()
#print("bin content sum = %d" % result)
#pr.print_stats()








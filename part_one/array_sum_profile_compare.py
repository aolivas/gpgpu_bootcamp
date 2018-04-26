#!/usr/bin/env python

import numpy as np
from numba import cuda

@cuda.jit
def sum_array(result, data):
    idx = cuda.threadIdx.x
    array = data[idx]
    for value in array:
        result[idx] += value

n_arrays = 1024
array_size = 10000
np.random.seed(42)
data = np.array([np.random.random(array_size)
                 for i in range(n_arrays)])

result1 = np.zeros(n_arrays)
sum_array[1, n_arrays](result1, data)
print(sum(result1))

import cProfile
pr = cProfile.Profile()
pr.enable()
result = np.sum(data)
pr.disable()
print("bin content sum = %d" % result)
pr.print_stats()








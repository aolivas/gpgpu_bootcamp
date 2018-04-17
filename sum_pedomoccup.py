#!/usr/bin/env python

import cPickle as pickle
import numpy as np

histograms = pickle.load(open('pedomoccup.pkl','r'))
harray = np.array([np.array(h['bin_values'])
                      for h in histograms])

from numba import cuda

@cuda.jit
def calc(result, harray):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx + ty * bw 
    arr = harray[idx]
    for bv in arr:
        result[idx] += bv

result = np.zeros(len(harray))
cuda.profile_start()
n_threads_per_block = 32
nblocks = (len(harray) + (n_threads_per_block - 1)) // n_threads_per_block
calc[nblocks, n_threads_per_block](result, harray)
cuda.profile_stop()
print(sum(result))
print("nblocks = %d" % nblocks)
print("nthreads = %d" % (nblocks * n_threads_per_block))



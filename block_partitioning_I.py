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
    arr = harray[tx]
    for bv in arr:
        result[tx] += bv

result = np.zeros(len(harray))
cuda.profile_start()
n_threads_per_block = len(harray)
nblocks = 1
calc[nblocks, n_threads_per_block](result, harray)
cuda.profile_stop()
print(sum(result))

print("nblocks = %d" % nblocks)
print("nthreads = %d" % (nblocks * n_threads_per_block))



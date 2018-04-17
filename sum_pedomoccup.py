#!/usr/bin/env python

import cPickle as pickle
import numpy as np

histograms = pickle.load(open('pedomoccup.pkl','r'))
harray = np.array([np.array(h['bin_values'])
                      for h in histograms])

from numba import cuda

# 1) kernels cannot return a value
# 2) kernels explicitly declare their thread hierarchy when called
@cuda.jit
def calc(result, harray):
    idx = cuda.threadIdx.x
    arr = harray[idx]
    for bv in arr:
        result[idx] += bv

result = np.zeros(len(harray))
cuda.profile_start()
calc[1,len(harray)](result, harray)
cuda.profile_stop()
print(sum(result))

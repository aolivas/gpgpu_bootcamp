#!/usr/bin/env python

from __future__ import print_function

import cProfile
import cPickle as pickle
import numpy

histograms = pickle.load(open('dbdump.pkl','r'))

test_histograms_names = histograms['test'][0]['histograms'].keys()
benchmark_histograms_names = histograms['benchmark'][0]['histograms'].keys()

hname = 'PEDOMOccup'
generator = 'tev_starting_muon'
hs = [h for n,h in histograms['test'][0]['histograms'].iteritems() if n == hname]
for result in histograms['benchmark']:
    if result['generator'] == generator:
        for n,h in result['histograms'].iteritems():
            if n == hname:
                hs.append(h)

harray = numpy.array([numpy.array(h['bin_values']) for h in hs])

from numba import cuda

print(len(harray))
@cuda.jit
def calc(result, harray):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    a = harray[pos]
    for e in a:
        result[pos] += e

results = numpy.zeros(len(harray))

gpu = cuda.get_current_device()
#threads_per_block = len(harray) # NB: MAX_THREADS_PER_BLOCK = 1024
#threads_per_block = 1024 # NB: MAX_THREADS_PER_BLOCK = 1024
### Let's make this hardware independent
threads_per_block = gpu.MAX_THREADS_PER_BLOCK
# if you choose an invalid value, CUDA throws.
# and is hardware dependent.
blocks_per_grid = (harray.size + (threads_per_block - 1)) // threads_per_block
print('Threads per block = %d' % threads_per_block)
print('Blocks per grid = %d' % blocks_per_grid)
print("Total number of threads launched = %d" % threads_per_block * blocks_per_grid)

cuda.profile_start()
calc[blocks_per_grid, threads_per_block](results, harray)
cuda.profile_stop()
# this should break down at some point, like when
# len(harray) exceeds the number of threads I can launch in a block.
# What's the maximum number of threads in a block?

print(len(results))
print(results)

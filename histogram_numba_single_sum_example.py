#!/usr/bin/env python

from __future__ import print_function

import cProfile
import cPickle as pickle
import pylab, numpy

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

#@cuda.jit(device = True)
#def norm_chisq(h1, h2):
#    return sum([(u - v)**2/(u + v)
#                for u,v in zip(h1, h2)
#                if u > 0 or v > 0])

# 1) kernels cannot return a value
# 2) kernels explicitly declare their thread hierarchy when called
#  Is this block or cyclic partitioning?
#  If it's cyclic
print(len(harray))
@cuda.jit
def calc(result, harray):
    print(len(harray))
    for e in harray:
        result[0] += e
    #results[0] = sum(harray)

# results = list() ### doesn't work
results = numpy.array([0])
results[0] = 0
#cProfile.run('calc[1,1](results, harray[0])')
cuda.profile_start()
calc[1,1](results, harray[0])
cuda.profile_stop()
print(len(results))
print(results[0])

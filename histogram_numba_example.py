#!/usr/bin/env python

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

#def norm_chisq(h1, h2):
#    return sum([(u - v)**2/(u + v)
#                for u,v in zip(h1, h2)
#                if u > 0 or v > 0])
#
#def calc(result, harray):
#    nhists = len(harray)
#    for i in range(nhists):
#        for j in range(i+1, nhists):
#            results.append(norm_chisq(harray[i], harray[j]))

from numba import cuda

@cuda.jit(device = True)
def norm_chisq(h1, h2):
    return sum([(u - v)**2/(u + v)
                for u,v in zip(h1, h2)
                if u > 0 or v > 0])

# 1) kernels cannot return a value
# 2) kernels explicitly declare their thread hierarchy when called
@cuda.jit
def calc(result, harray):
    i = ???
    j = ???
    results.append(norm_chisq(harray[i], harray[j]))

results = list()
cProfile.run('calc(results, harray)')
print(len(results))


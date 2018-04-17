#!/usr/bin/env python

import cPickle as pickle
import numpy as np

histograms = pickle.load(open('pedomoccup.pkl','r'))
harray = np.array([np.array(h['bin_values'])
                   for h in histograms])

import cProfile 
pr = cProfile.Profile()
pr.enable()
result = np.sum(harray)
pr.disable()
print("bin content sum = %d" % result)
pr.print_stats()



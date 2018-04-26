#!/usr/bin/env python

from __future__ import print_function

import numpy as np

N = 8192
a = np.random.random((N,N))
b = np.random.random((N,N))

def add(a,b):
    return a + b

import cProfile 
pr = cProfile.Profile()
print("Start profiling...")
pr.enable()
r = add(a,b)
pr.disable()
print("...stop profiling.")
print(r)
pr.print_stats()




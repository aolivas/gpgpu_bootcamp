#!/usr/bin/env python

import numpy as np
from numba import cuda, float64

N = 16 # multiply two NxN matrices

@cuda.jit
def slow_multiply(A, B, C):
    x,y = cuda.grid(2)
    tmp = 0.
    for k in range(N):
        tmp += A[x,k] * B[k,y]
    C[x,y] = tmp

@cuda.jit
def fast_multiply(A, B, C):
    # assume this happens once per block, not per grid.
    sA = cuda.shared.array(shape=(N,N), dtype = float64)
    sB = cuda.shared.array(shape=(N,N), dtype = float64)

    x,y = cuda.grid(2) # absolute position in the grid
    sA[x, y] = A[x, y] # load an element into shared memory           
    sB[x, y] = B[x, y] # load an element into shared memory
    cuda.syncthreads() # wait for threads to load their data
        
    tmp = 0.
    for j in range(N):
        tmp += sA[x, j] * sB[j, y]
    C[x,y] = tmp

np.random.seed(42)
A = np.random.random((N,N))
B = np.random.random((N,N))
fast_C = np.zeros((N,N))
slow_C= np.zeros((N,N))

# burn one
c = np.zeros((N,N))
fast_multiply[1,1](A,B,c)

import sys
import time
cuda.profile_start()
if len(sys.argv) > 1:
    start_time = time.time()
    slow_multiply[1,(N,N)](A, B, slow_C)
    end_time = time.time()
else:
    start_time = time.time()
    fast_multiply[1,(N,N)](A, B, fast_C)
    end_time = time.time()
cuda.profile_stop()
print("elapsed time = %.2f ms" % (1e3*(end_time - start_time)))
#print("elapsed time (fast) = %.2f ms" % (1e3*(end_time - mid_time)))
#assert(np.array_equal(fast_C, slow_C))


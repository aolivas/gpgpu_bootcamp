#!/usr/bin/env python

from __future__ import print_function

from numba import cuda

@cuda.jit(device=True)
def hello_world():
    print("Hello World")

@cuda.jit(debug=True)
def kernel():
    hello_world()

#kernel launch
cuda.profile_start()
kernel[1,5]()
cuda.profile_stop()


#!/usr/bin/env python

from __future__ import print_function

from numba import cuda

@cuda.jit(debug=True)
def kernel():
    print("Hello World")

#kernel launch
kernel[1,5]()





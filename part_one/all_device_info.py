#!/usr/bin/env python

from __future__ import print_function

from numba import cuda
 
device_attributes = [
    "name",
    "id",
    "compute_capability",
    "MAX_THREADS_PER_BLOCK",
    "MAX_BLOCK_DIM_X", 
    "MAX_BLOCK_DIM_Y", 
    "MAX_BLOCK_DIM_Z", 
    "MAX_GRID_DIM_X", 
    "MAX_GRID_DIM_Y", 
    "MAX_GRID_DIM_Z",
    "MAX_SHARED_MEMORY_PER_BLOCK", 
    "ASYNC_ENGINE_COUNT",
    "CAN_MAP_HOST_MEMORY", 
    "MULTIPROCESSOR_COUNT",
    "WARP_SIZE", 
    "UNIFIED_ADDRESSING", 
    "PCI_BUS_ID", 
    "PCI_DEVICE_ID" 
]

for gpu in cuda.devices.gpus: 
    for attr in device_attributes:
        print("%s: %s" % (attr, str(getattr(gpu, attr))))



     

#!/usr/bin/env python

import pycuda.autoinit
import pycuda.driver as driver

print("%d device%s found." % (driver.Device.count(),
                              "" if driver.Device.count() == 1 else "s"))

for device_number in range(driver.Device.count()):
    device = driver.Device(device_number)
    print("Device #%d: %s" % (device_number, device.name()))
    print("Compute Capability: %d.%d" % device.compute_capability())
    print("Total Memory: %s KB" % (device.total_memory()//1024))



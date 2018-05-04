#!/usr/bin/env python

import sys
import numpy as np

def fit(pulses):
    avg_time = 0.
    avg_time_sq = 0.
    avg_pos = np.zeros(3)
    avg_tp = np.zeros(3)
    
    qsum = 0
    for key, (pos, hits) in pulses.iteritems():
        for q,t in hits:
            qsum += q
            avg_time += q*t
            avg_time_sq += q*t*t
            avg_pos += q * pos
            avg_tp += q * t * pos 

    avg_pos /= qsum
    avg_tp /= qsum
    avg_time /= qsum       
    avg_time_sq /= qsum       

    velocity = (avg_tp - avg_pos * avg_time)/(avg_time_sq - avg_time**2)    
    return velocity
    
import cPickle as pickle
events = pickle.load(open(sys.argv[1]))

for event in events:
    lf = fit(event['pulses'])
    print(lf)

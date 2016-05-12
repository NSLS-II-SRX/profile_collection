# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:10:28 2016

@author: xf05id1
"""
from time import sleep as tsleep

print('1. testing individual motor motion:')
#wriite smarter way to decide the inital 'neutral position'
print('Move dcm bragg to a neutral position (5 deg) away from the limits.')
mov(dcm.bragg,5)

print('Testing relative motion within motion range')
print('- Moving +1 degree..')
movr(dcm.bragg,1)
print('- Moving -1 degree..')
movr(dcm.bragg,-1)

#should raise an exception
#print('Testing behavior when motor hits the limit')
#mov(dcm.bragg,-5)

print('Testing emergency stop, pressing Ctrl+C within 5 s.')
try:
    movr(dcm.bragg, 5)
except KeyboardInterrupt:
    print('canceled', dcm.bragg.position)
else:
    raise RuntimeError("YOU DID NOT ABORT THE MOVE")
print('shortly after canceled', dcm.bragg.position)
tsleep(1)
print('1 s later', dcm.bragg.position)
tsleep(1)
print('2 s later', dcm.bragg.position)

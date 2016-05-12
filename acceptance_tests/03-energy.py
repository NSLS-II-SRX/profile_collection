# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:41:58 2016

@author: xf05id1
"""

print('2. testing pseudo motors motion:')

print('testing energy')
print('Testing relative motion within motion range')
print('- Moving +1 keV..')
movr(energy.energy, 1)
print('- Moving -1 keV..')
movr(energy.energy, -1)

#for complete test, checking all individual motors (bragg, c2x, and ugap are stopped)
print('Testing emergency stop, pressing Ctrl+C within 5 s.')
try:
    movr(energy.energy, 5)
except KeyboardInterrupt:
    print('canceled', energy.position.energy)
else:
    raise RuntimeError("YOU DID NOT ABORT THE MOVE")
print('shortly after canceled', energy.position.energy)
tsleep(1)
print('1 s later', energy.position.energy)
tsleep(1)
print('2 s later', energy.position.energy)


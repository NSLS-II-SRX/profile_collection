# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:41:58 2016

@author: xf05id1
"""

from time import sleep as tsleep
print('2. testing pseudo motors motion:')

print('testing ssa')
print('Testing relative motion within motion range')
print('- Moving h_gap +0.1 mm')
movr(slt_ssa.h_gap, .1)
print('- Moving h_gap -0.1 mm..')
movr(slt_ssa.h_gap, -.1)

#for complete test, checking all h_gap, h_cen, v_gap, v_cen motions
print('Testing emergency stop, pressing Ctrl+C within 1 s.')
try:
    movr(slt_ssa.h_gap, 5)
except KeyboardInterrupt:
    print('canceled', slt_ssa.h_gap.position)
else:
    raise RuntimeError("YOU DID NOT ABORT THE MOVE")
print('shortly after canceled', slt_ssa.h_gap.position)
tsleep(1)
print('1 s later', slt_ssa.h_gap.position)
tsleep(1)
print('2 s later', slt_ssa.h_gap.position)

# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:10:28 2016

@author: xf05id1
"""
from time import sleep as tsleep
from bluesky.suspenders import SuspendBoolLow

print('6. testing suspenders:')
gv = EpicsSignal('XF:05IDB-VA:1{Slt:SSA-GV:1}Pos-Sts')
susp_gv = SuspendBoolLow(gv)
RE.install_suspender(susp_gv)

print('going to launch xrf2d map, please ensure GV before B shutter is closed')
tsleep(5)

print('running xrf 2d mapping, open GV to test start of the scan, close GV to test suspension, re-open GV to test resuming the scan')
hf2dxrf(xstart=26.70,xstepsize=0.005,xnumstep=3,ystart=15.28,ystepsize=.005,ynumstep=3,acqtime=.2,numrois=2, i0map_show=True, itmap_show=True)


print('going to launch xanes scan, please ensure GV before B shutter is closed')
tsleep(5)

print('running xanes, open GV to test start of the scan, close GV to test suspension, re-open GV to test resuming the scan')
xanes(erange = [7112-30, 7112-20, 7112+30], 
            estep = [2, 5],  
            harmonic = None,            
            acqtime=0.2, roinum=1, i0scale = 1e8, itscale = 1e8,samplename='test',filename='test')


# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:10:28 2016

@author: xf05id1
"""
from time import sleep as tsleep

print('5. testing integrated scan functions:')

print('testing xrf 2d mapping')
hf2dxrf(xstart=26.70,xstepsize=0.005,xnumstep=3,ystart=15.28,ystepsize=.005,ynumstep=3,acqtime=.2,numrois=2, i0map_show=True, itmap_show=True)

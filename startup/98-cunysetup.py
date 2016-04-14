# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:26:52 2016

@author: xf05id1
"""

#Mn_xoffset = 24.674504759503112
#Cu_xoffset = 24.643593546692397
#Bi_xoffset = 24.721333099499144
#
#Mn_c1r = -4.864
#Cu_c1r = -4.804
#Bi_c1r = -4.764
#
#MnK = 6539
#CuK = 8979
#BiL3 = 13419
#
#sample = '1stsample'
#
#def changeE(setpoint):
#    if setpoint is 'Mn':
#        energy._xoffset = Mn_xoffset
#        dcm.c1_roll.set(Mn_c1r)                
#        energy.set(6.6)
#    elif setpoint is 'Cu':
#        energy._xoffset = Cu_xoffset
#        dcm.c1_roll.set(Cu_c1r)
#        energy.set(9.1)
#    elif setpoint is 'Bi':
#        energy._xoffset = Bi_xoffset
#        dcm.c1_roll.set(Bi_c1r)
#        energy.set(14.0)
#    elif setpoint is 'XRF':
#        energy._xoffset = Bi_xoffset
#        dcm.c1_roll.set(Bi_c1r)
#        energy.set(15.0)
#    else:
#        print("wrong setting, please set it to 'Mn', 'Cu', 'Bi', or 'XRF'")
#
#
#def cunyexp():
#
#    #run Mn XANES
#    xanes(erange = [MnK-50, MnK-20, MnK+50, MnK+120], 
#            estep = [2, 1, 5],  
#            harmonic = None, correct_c2_x = Mn_xoffset, correct_c1_r = Mn_c1r,             
#            acqtime=None, roinum=1, i0scale = 1e8, itscale = 1e8,
#            samplename = sample, filename = sample)
#            
#    #run Cu XANES
#    #run Bi XANES
#    
#    #XRF mapping?
#    
#    #run Bi XANES
#    #run Cu XANES
#    #run Mn XANES
#
#    
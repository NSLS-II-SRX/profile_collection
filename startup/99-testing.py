# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:22:22 2016

@author: xf05id1
"""

from bluesky.plans import scan, list_scan, abs_set
from bluesky.callbacks import LiveTable


def basic_scan():
    print('start basic scan')
    basic_scan_plan = scan([], tomo_stage.finex_top, 0, 10, 5)
    basic_scan_gen = yield from basic_scan_plan
    print('done with basic scan')
    
    return basic_scan_gen

def tomo_test(thetastart = 90, thetastop = 80, thetanumstep = 3):
    theta_traj = np.linspace(thetastart, thetastop, thetanumstep)
    
    for theta_setpt in theta_traj:
        print('current angle')
        print(tomo_stage.theta.position)
        print('move angle to '+str(theta_setpt))
#        tomo_lab_z_initi = yield from abs_set(tomo_lab.lab_z, 0) this does not work
        tomo_lab_z_initi = yield from list_scan([], tomo_lab.lab_z, [0])
        tomo_theta_set_gen = yield from list_scan([tomo_stage], tomo_stage.theta, [theta_setpt])
        print('start running basic_scan')
        basic_scan_gen = yield from basic_scan() 
        print('done running basic scan')
        
    
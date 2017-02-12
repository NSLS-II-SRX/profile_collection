# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:48:07 2016

@author: xf05id1
"""

from bluesky.plans import AbsScanPlan
from bluesky.callbacks.scientific import PeakStats, plot_peak_stats
import matplotlib.pylab as plt
import time


def bpmAD_exposuretime_adjust():  
    maxct = bpmAD.stats1.max_value.get()
    if maxct < 150:
        while(bpmAD.stats1.max_value.get() <= 170):
            current_exptime = bpmAD.cam.acquire_time.value
            bpmAD.cam.acquire_time.set(current_exptime+0.0005)
            time.sleep(0.5)
    elif maxct > 170:
        while(bpmAD.stats1.max_value.get() >= 150):
            current_exptime = bpmAD.cam.acquire_time.value
            bpmAD.cam.acquire_time.set(current_exptime-0.0005)
            time.sleep(0.5)

def undulator_calibration(outfile = 'SRXUgapCalibration.txt', u_gap_start = 9.53, u_gap_end = 18.03, u_gap_step = 0.5):

    '''
    outfile (string) is a .txt file for the lookup table, desirable to name it with the date of the calibration, e.g. SRXUgapCalibration20161225.txt
    undulator gap set point range are defined in:
        u_gap_start (float)
        u_gap_end (float)
        u_gap_step (float)
    '''  
    
    bpmAD.cam.read_attrs = ['acquire_time']
    bpmAD.configuration_attrs = ['cam']
    
    #default output directory for the lookup table
    UCalibDir = '/nfs/xf05id1/UndulatorCalibration/'
    
    newfile = False
    
    if newfile is True:
        f = open(UCalibDir+outfile, 'w')
        f.write('undulator_gap fundemental_energy\n')
        f.close()
    
    
    #bragg scan setup default
    energy_res = 0.002 #keV
    bragg_scanwidth = 0.1 #keV
    bragg_scanpoint = bragg_scanwidth*2/energy_res+1 
    harmonic = 3

    energy.harmonic.set(harmonic)
    
    #generate lookup table by scanning Bragg at each undulator gap set point
    for u_gap_setpoint in numpy.arange(u_gap_start, u_gap_end+u_gap_step, u_gap_step):
    
        energy_setpoint = float(energy.u_gap.utoelookup(u_gap_setpoint))*harmonic
        
        print('move u_gap to:', u_gap_setpoint)
        print('move bragg energy to:', energy_setpoint)
        
        #energy.move_c2_x.put(True)
        energy.move_u_gap.set(True)
        time.sleep(0.2)    
        energy.move(energy_setpoint)
        ps = PeakStats(energy.energy.name, bpmAD.stats1.total.name)
        
        bpmAD_exposuretime_adjust()    
        
        energy.move_u_gap.set(False)
        braggscan=AbsScanPlan([bpmAD, pu, ring_current], energy, energy_setpoint-bragg_scanwidth, energy_setpoint+bragg_scanwidth, bragg_scanpoint)
        liveploty = bpmAD.stats1.total.name
        livetableitem = [energy.energy, bpmAD.stats1.total, ring_current]
        liveplotx = energy.energy.name
        liveplotfig1 = plt.figure()
        plt.show()
        
        gs.RE(braggscan, [LiveTable(livetableitem),                      
                          LivePlot(liveploty, x=liveplotx, fig=liveplotfig1),
                          ps])
                          
        maxenergy = ps.max[0]
        maxintensity = ps.max[1]
        fwhm = ps.fwhm
        print('max energy is:', maxenergy)
        print('fundemental energy:', maxenergy/harmonic)
        
        f = open(UCalibDir+outfile, 'a')
        f.write(str(energy.u_gap.position)+' '+str(maxenergy/harmonic)+'\n')
        f.close()

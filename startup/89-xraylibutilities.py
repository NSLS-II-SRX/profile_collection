# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:38:05 2016

@author: xf05id1
"""

import xraylib
def setroi(element = None, line = 'Ka', roisize = 200, roi=1):
    '''
    setting rois for Xspress3 by providing elements
    input:
        element (string): element of interest, e.g. 'Se'
        line (string): default 'Ka', options:'Ka', 'Kb', 'La', 'Lb', 'M', Ka1', 'Ka2', 'Kb1', 'Kb2', 'Lb2', 'Ma1'
        roisize (int): roi window size in the unit of eV, default = 200
        roi (int): 1, 2, or 3: the roi number to set, default = 1       
        
    '''   
    
    atomic_num = xraylib.SymbolToAtomicNumber(element)
    multilineroi_flag = False
    print('element:', element)
    print('roi window size (eV):', roisize)
    
    #calculate the roi low and high bound, based on the line
    if line is 'Ka': #using average of Ka1 and Ka2 as the center of the energy/roi
        line_h = xraylib.LineEnergy(atomic_num, xraylib.KL3_LINE)*1000 #Ka1
        line_l = xraylib.LineEnergy(atomic_num, xraylib.KL2_LINE)*1000 #Ka2
        print('Ka1 line (eV):', line_h)
        print('Ka2 line (eV):', line_l)                 
        energy_cen = (line_l+line_h)/2
        multilineroi_flag = True       
    elif line is 'Kb': #using Kb1 line only as the center of the energy/roi
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KM3_LINE)*1000      
        print('Kb1 line (eV):', energy_cen)   
    elif line is 'La': #using average of La1 and La2 as the center of the energy/roi
        line_h = xraylib.LineEnergy(atomic_num, xraylib.L3M5_LINE)*1000 #La1
        line_l = xraylib.LineEnergy(atomic_num, xraylib.L3M4_LINE)*1000 #La2
        print('La1 line (eV):', line_h)
        print('La2 line (eV):', line_l) 
        energy_cen = (line_l+line_h)/2 
        multilineroi_flag = True           
    elif line is 'Lb':  #using average of Lb1 and Lb2 as the center of the energy/roi
        line_l = xraylib.LineEnergy(atomic_num, xraylib.L2M4_LINE)*1000 #Lb1
        line_h = xraylib.LineEnergy(atomic_num, xraylib.L3N5_LINE)*1000 #Lb2
        print('Lb2 line (eV):', line_h) 
        print('Lb1 line (eV):', line_l)
        energy_cen = (line_l+line_h)/2 
        multilineroi_flag = True
    elif line is 'M': #using Ma1 line only as the center of the energy/roi
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.M5N7_LINE)*1000        
        print('Ma1 line (eV):', energy_cen)
    elif line is 'Ka1':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KL3_LINE)*1000        
        print('Ka1 line (eV):', energy_cen)
    elif line is 'Ka2':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KL2_LINE)*1000        
        print('Ka2 line (eV):', energy_cen)
    elif line is 'Kb1':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.KM3_LINE)*1000
        print('Kb1 line (eV):', energy_cen)
    elif line is 'Lb1':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.L2M4_LINE)*1000
        print('Kb2 line (eV):', energy_cen)
    elif line is 'Lb2':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.L3N5_LINE)*1000
        print('Lb2 line (eV):', energy_cen)
    elif line is 'Ma1':
        energy_cen = xraylib.LineEnergy(atomic_num, xraylib.M5N7_LINE)*1000 
        print('Ma1 line (eV):', energy_cen)
        
    print('energy center (eV):', energy_cen)

    roi_cen = energy_cen/10.  #converting energy center position from keV to eV, then to channel number 
    roi_l = round(roi_cen - roisize/10/2)
    roi_h = round(roi_cen + roisize/10/2)

    print('roi center:', roi_cen)
    print('roi lower bound:', roi_l, ' (', roi_l*10, ' eV )')
    print('roi higher bound:', roi_h, ' (', roi_h*10, ' eV )')


    if roi_l <= 0:
        raise Exception('Lower roi bound is at or less than zero.')
    if roi_h >= 2048:
        raise Exception('Higher roi bound is at or larger than 2048.')
    
    if multilineroi_flag is True:
        print('lowest emission line to roi lower bound:', line_l - roi_l*10, 'eV')
        print('highest emission line to roi higher bound:', line_h - roi_h*10, 'eV')
        
        if roi_l*10 - line_l > 0:
            print('Warning: window does not cover the lower emission line. Consider making roisize larger.\n',
                   'currently the window lower bound is higher than lower emission line by ', roi_l*10 - line_l, 'eV')
        if line_h - roi_h*10 > 0:
            print('Warning: window does not cover the higher emission line. Consider making roisize larger.\n',
                   'currently the window higher bound is less than higher emission line by ', line_h - roi_h*10, 'eV')
            
    #set up roi values       
    if roi is 1:                
        xs.channel1.rois.roi01.bin_low.set(roi_l)
        xs.channel1.rois.roi01.bin_high.set(roi_h)
        xs.channel2.rois.roi01.bin_low.set(roi_l)
        xs.channel2.rois.roi01.bin_high.set(roi_h)
        xs.channel3.rois.roi01.bin_low.set(roi_l)
        xs.channel3.rois.roi01.bin_high.set(roi_h)  
    elif roi is 2:                
        xs.channel1.rois.roi02.bin_low.set(roi_l)
        xs.channel1.rois.roi02.bin_high.set(roi_h)
        xs.channel2.rois.roi02.bin_low.set(roi_l)
        xs.channel2.rois.roi02.bin_high.set(roi_h)
        xs.channel3.rois.roi02.bin_low.set(roi_l)
        xs.channel3.rois.roi02.bin_high.set(roi_h)
    elif roi is 3:                
        xs.channel1.rois.roi03.bin_low.set(roi_l)
        xs.channel1.rois.roi03.bin_high.set(roi_h)
        xs.channel2.rois.roi03.bin_low.set(roi_l)
        xs.channel2.rois.roi03.bin_high.set(roi_h)
        xs.channel3.rois.roi03.bin_low.set(roi_l)
        xs.channel3.rois.roi03.bin_high.set(roi_h)
    else:
        print('cannot set roi values; roi = 1, 2, or 3')
        
def edge(element = None, line = 'K', unit = 'eV'):
    '''
    function return edge (K or L3) in eV or keV with input element sympbol
    '''
    
    atomic_num = xraylib.SymbolToAtomicNumber(element)
        
    if line is 'K':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.K_SHELL)
    if line is 'L3':
        edge_value = xraylib.EdgeEnergy(atomic_num, xraylib.L3_SHELL)        
    
    if unit is 'eV':
        return edge_value*1000
    else:
        return edge_value
    
    
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
        line (string): 'Ka' for now
        roisize (int): roi window size in the unit of eV, default = 200
        roi (int): 1, 2, or 3: the roi number to set, default = 1       
        
    '''   
    
    atomic_num = xraylib.SymbolToAtomicNumber(element)
    
    #calculate the roi low and high bound, based on the line
    if line is 'Ka':
        Ka1 = xraylib.LineEnergy(atomic_num, xraylib.KL3_LINE)
        Ka2 = xraylib.LineEnergy(atomic_num, xraylib.KL2_LINE)
        roi_cen = (Ka1+Ka2)*1000/10/2.
        roi_l = roi_cen - roisize/10/2
        roi_h = roi_cen + roisize/10/2
 
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
    
    
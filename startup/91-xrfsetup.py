# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:16:52 2016

set up for 2D XRF scan for HR mode
@author: xf05id1
"""

#TO-DOs:
    #1. add suspender once it's fixed
    #2. put snake back to the scan once it's fixed
    #3. check the x/y are correct
    #4. put x/y axes onto the live plot
    #5. add i0 into the default figure

from bluesky.plans import OuterProductAbsScanPlan
from bluesky.callbacks import LiveRaster
import matplotlib

#matplotlib.pyplot.ticklabel_format(style='plain')

def hf2dxrf(xstart=None, xnumstep=None, xstepsize=None, 
            ystart=None, ynumstep=None, ystepsize=None, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=None, numrois=1):

    '''
    example: 
    '''

    #make sure user provided correct input

    if xstart is None:
        raise Exception('xstart = None, must specify an xstart position')
    if xnumstep is None:
        raise Exception('xnumstep = None, must specify an xnumstep position')
    if xstepsize is None:
        raise Exception('xstepsize = None, must specify an xstepsize position')
    if ystart is None:
        raise Exception('ystart = None, must specify an ystart position')
    if ynumstep is None:
        raise Exception('ynumstep = None, must specify an ynumstep position')
    if ystepsize is None:
        raise Exception('ystepsize = None, must specify an ystepsize position')
    if acqtime is None:
        raise Exception('acqtime = None, must specify an acqtime position')

    #setup the detector
    current_preamp.exp_time.put(acqtime)
    saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    for roi_idx in range(numrois):
        saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.net_count')
        saturn.read_attrs.append('mca.rois.roi'+str(roi_idx)+'.count')
       
    det = [current_preamp, saturn]        

    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [hf_stage.x, hf_stage.y, 'current_preamp_ch0', 'current_preamp_ch1']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    for roi_idx in range(numrois):
        livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_net_count')
        livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_count')
        #roimap = LiveRaster((xnumstep, ynumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_net_count', clim=None, cmap='viridis', xlabel='x', ylabel='y', extent=None)
        colormap = 'jet' #previous set = 'viridis'
        roimap = LiveRaster((xnumstep, ynumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_count', clim=None, cmap='jet', 
                            xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
        livecallbacks.append(roimap)

    livecallbacks.append(LiveTable(livetableitem, max_post_decimal = 4)) 

    
    #setup the plan  
    #OuterProductAbsScanPlan(detectors, *args, pre_run=None, post_run=None)
    #OuterProductAbsScanPlan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)
    hf2dxrf_scanplan = OuterProductAbsScanPlan(det, hf_stage.y, ystart, ystop, ynumstep, hf_stage.x, xstart, xstop, xnumstep, False)
    scaninfo = gs.RE(hf2dxrf_scanplan, livecallbacks)

    #write to scan log    
    logscan('2dxrf')    
    
    return scaninfo

    
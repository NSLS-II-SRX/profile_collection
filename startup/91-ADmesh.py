# -*- coding: utf-8 -*-
"""
set up for 2D mesh scan for HF station using area detector

Created on Fri Mar 4 2016

@author: gjw
"""

from bluesky.plans import OuterProductAbsScanPlan
from bluesky.callbacks import LiveRaster
import matplotlib


def hf2dad(xstart=None, xnumstep=None, xstepsize=None, 
            ystart=None, ynumstep=None, ystepsize=None, 
            #wait=None, simulate=False, checkbeam = False, checkcryo = False, #need to add these features
            acqtime=None, num_images=1):

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

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    coherent_dict = {'hfm_bend':hfm.bend.position}
    gs.RE.md['beamline_status'].update(coherent_dict)
    gs.RE.md['AD_params'] = {'aquire_time':pixi.det.acquire_time.get(),
        'num_images':pixi.det.num_images.get(),
        'temperature_set':pixi.det.temperature.get(),
        'temperature_act':pixi.det.temperature_actual.get(),
        'humidity':pixi.det.humidity_box.get(),
        'HV_act':pixi.det.hv_actual.get()
        }
    metadata_record()

    #setup the detector
    #think this is the wrong approach... should be calling configure()...
    current_preamp.exp_time.put(acqtime*.8)
    pixi.det.acquire_time.put(acqtime)
    pixi.det.num_images.put(num_images)

    det = [current_preamp,pixi]        


    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = [hf_stage.x, hf_stage.y, 'current_preamp_ch0', 'current_preamp_ch2']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    print('xstop = '+str(xstop))  
    print('ystop = '+str(ystop)) 
    
    
    #for roi_idx in range(numrois):
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_net_count')
    #    livetableitem.append('saturn_mca_rois_roi'+str(roi_idx)+'_count')
    #    #roimap = LiveRaster((xnumstep, ynumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_net_count', clim=None, cmap='viridis', xlabel='x', ylabel='y', extent=None)
    #    colormap = 'jet' #previous set = 'viridis'
    #    roimap = LiveRaster((ynumstep, xnumstep), 'saturn_mca_rois_roi'+str(roi_idx)+'_count', clim=None, cmap='jet', 
    #                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
    #    livecallbacks.append(roimap)


    i0map = LiveRaster((ynumstep, xnumstep), 'current_preamp_ch2', clim=None, cmap='jet', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
    livecallbacks.append(i0map)


#    commented out liveTable in 2D scan for now until the prolonged time issue is resolved
    livecallbacks.append(LiveTable(livetableitem)) 

    
    #setup the plan  
    #OuterProductAbsScanPlan(detectors, *args, pre_run=None, post_run=None)
    #OuterProductAbsScanPlan(detectors, motor1, start1, stop1, num1, motor2, start2, stop2, num2, snake2, pre_run=None, post_run=None)
    hf2dxrf_scanplan = OuterProductAbsScanPlan(det, hf_stage.y, ystart, ystop, ynumstep, hf_stage.x, xstart, xstop, xnumstep, True)
    scaninfo = gs.RE(hf2dxrf_scanplan, livecallbacks)

    #write to scan log    
    logscan('2dxrf')    
    
    return scaninfo

    

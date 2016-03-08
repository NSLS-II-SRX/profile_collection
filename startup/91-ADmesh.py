# -*- coding: utf-8 -*-
"""
set up for 2D mesh scan for HF station using area detector

Created on Fri Mar 4 2016

@author: gjw
"""

from bluesky.plans import OuterProductAbsScanPlan
from bluesky.callbacks import LiveRaster
from bluesky.broker_callbacks import LiveImage
from databroker import DataBroker as db
import matplotlib
import epics


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
    metadata_record()
    coherent_dict = {'hfm_bend':hfm.bend.position}
    gs.RE.md['beamline_status'].update(coherent_dict)
    gs.RE.md['AD_params'] = {'aquire_time':pixi.det.acquire_time.get(),
        'num_images':pixi.det.num_images.get(),
        'temperature_set':pixi.det.temperature.get(),
        'temperature_act':pixi.det.temperature_actual.get(),
        'humidity':pixi.det.humidity_box.get(),
        'HV_act':pixi.det.hv_actual.get()
        }
    gs.RE.md['scan_params'] = {'xstart':xstart,
        'xnumstep':xnumstep,
        'xstepsize':xstepsize,
        'ystart':ystart,
        'ynumstep':ynumstep,
        'ystepsize':ystepsize,
        'acqtime':acqtime,
        'num_images':num_images,
        'zpos':hf_stage.z.position
        }

    #setup the detector
    #think this is the wrong approach... should be calling configure()...
    current_preamp.exp_time.put(acqtime*.8)
    pixi.det.acquire_time.put(acqtime)
    pixi.det.num_images.put(num_images)
    pixi.tiff.auto_save.put(0)
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1)

    det = [current_preamp,pixi]        

    #setup the live callbacks
    livecallbacks = []
    
    livetableitem = ['hf_stage_x', 'hf_stage_y', 'current_preamp_ch2','pixi_stats1_sigma_x','pixi_stats1_sigma_y']

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize  
  
    first_map = LiveRaster((ynumstep, xnumstep), 'pixi_stats1_total', clim=None, cmap='inferno', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
    livecallbacks.append(first_map)
    second_map = LiveRaster((ynumstep, xnumstep), 'pixi_stats2_total', clim=None, cmap='inferno', 
                        xlabel='x (mm)', ylabel='y (mm)', extent=[xstart, xstop, ystop, ystart])
    livecallbacks.append(second_map)
    #this is causing seg faults
#    pixi_frame = SRXLiveImage('pixi_image')
#    livecallbacks.append(pixi_frame)

    livecallbacks.append(LiveTable(livetableitem)) 

    #setup the plan  
    hf2dad_scanplan = OuterProductAbsScanPlan(det, hf_stage.y, ystart, ystop, ynumstep, hf_stage.x, xstart, xstop, xnumstep, True)
    scaninfo = gs.RE(hf2dad_scanplan, livecallbacks)

    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)
    logscan_detailed('ADmesh')

    #dirty text output
    h=db[-1]
    fn = '/nfs/xf05id1/userdata/'+str(h['start']['proposal']['cycle'])+'/'+\
        str(h['start']['proposal']['saf_num'])+'_'+\
        str(h['start']['proposal']['PI_lastname'])+'/'+str(h['start']['scan_id'])+\
        '.log'
    fp=open(fn,'w')
    tab=get_table(h,['time','hf_stage_x','hf_stage_y'])
    fp.write(tab.to_csv())
    fp.close()

    return scaninfo


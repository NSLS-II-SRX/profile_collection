# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:13 2016

@author: xf05id1
"""

#for tomography
from bluesky.plans import subs_wrapper, scan, count, list_scan
import time
  
def tomo_fullfield(thetastart = -90, thetastop = 90, numproj = 361, ffwait = 2,
            acqtime = 0.002, record_preamp = True, preamp_acqtime = None,
            num_darkfield = 10, dfwait = 2,
            num_whitefield = 10, wf_sam_movrx = 0, wf_sam_movry = -1, wfwait = 2,
            eventlog_list = ['pcoedge_tiff_file_name']
            ):
                
    '''
    argurments:
    thetastart (float): angle for the start of the fullfied scan, in degree
    thetastop  (float): angle for the stop of the fullfied scan, includsive, in degree
    numproj (int): number of projections to collected within the angular range from thetastart to thetastop
    
    acqtime (float): acqusition time for pco.edge camera, in second
    record_preamp (boolean, default = True): when True, the current_preamp will be included in detecotr list in all data collection 
    preamp_acqtime (float): record_preamp is True, this value will be used to set the acqsition time for the current_preamp, 
                            if not defined, the same acqusition time as the pco.edge will be used, as set in acqtime
    
    num_darkfield  (int): number of dark field images collected before the scan; shutter will be atuomatically closed/opened before/after this collection
    num_whitefield (int): number of white field images collected before and after the scan, 
                          sample will be moved out of field of view prior to this collection, as defined in wf_sam_movrx and wf_sam_movry
    wf_sam_movrx (float): distance to move x before/after the tomography collection to collect white field
    wf_sam_movry (float): distance to move y before/after the tomography collection to collect white field
    eventlog_list (list of strings): fileds to record in the datalog file from evnets[0]['data']
    
    ffwait (float): time to wait before collecting full field tomography, in second
    wfwait (float): time to wait before collecting white field, in second
    d
    '''               
    
    print('start of full field tomography')
    print('acqusition time', acqtime)
 
    theta_traj = np.linspace(thetastart, thetastop, numproj)
    tomo_scan_output = []
    
    #setup detectors
    det = [pcoedge, tomo_stage.theta]
    pcoedge.cam.acquire_time.set(acqtime)
    
    if record_preamp is True:
        det.append(current_preamp)
        print('recording preamp')
        if preamp_acqtime is None:
            current_preamp.exp_time.set(acqtime)
            print('preamp acqtime', acqtime)
        else:
            current_preamp.exp_time.set(preamp_acqtime)
            print('preamp acqtime = ', preamp_acqtime)
    else:
        print('NOT recording preamp')
                    
    livecallbacks = []    
    livetableitem = ['tomo_stage_theta']
    if record_preamp is True:
        livetableitem.append('current_preamp_ch2')
    livecallbacks.append(LiveTable(livetableitem))
        
    #close the shutter
    print('closing shutter to collect darkfield')
    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)    
    #collecting darkfield
    time.sleep(dfwait)
    print('shutter closed, start collecting darkfield images, num = ', num_darkfield)
    fftomo_df_plan = count(det, num = num_darkfield)
    fftomo_df_plan = bp.subs_wrapper(fftomo_df_plan, livecallbacks)
    fftomo_df_gen = yield from fftomo_df_plan
    logscan_event0info('full_field_tomo_dark_field', event0info = eventlog_list)
    #logscan('full_field_tomo_dark_field')     
    print('darkfield collection done')
    
    #move sample out prior to white field collection
    print('moving sample out of field of view')
    print('moving sample x relative by', wf_sam_movrx)
    print('moving sample y relative by', wf_sam_movry)

    movesamplex_out = yield from list_scan([], tomo_stage.x, [tomo_stage.x.position+wf_sam_movrx])
    movesampley_out = yield from list_scan([], tomo_stage.y, [tomo_stage.y.position+wf_sam_movry])
    time.sleep(wfwait)
    print('sample out')
    
    #open the shutter
    print('opening shutter to collect whitefield')
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1) 

    #collecting whitefield
    print('shutter opened, start collecting whitefield images, num = ', num_whitefield)        
    fftomo_wf_plan = count(det, num = num_whitefield)
    fftomo_wf_plan = bp.subs_wrapper(fftomo_wf_plan, livecallbacks)
    fftomo_wf_gen = yield from fftomo_wf_plan
    logscan_event0info('full_field_tomo_white_field_prescan', event0info = eventlog_list)
    #logscan('full_field_tomo_white_field_prescan')      

    #move sample back after white field collection
    print('moving sample back to the field of view')
    print('moving sample x relative by', (-1)*wf_sam_movrx)
    print('moving sample y relative by', (-1)*wf_sam_movry)
    movesamplex_in = yield from list_scan([], tomo_stage.x, [tomo_stage.x.position-wf_sam_movrx])
    movesampley_in = yield from list_scan([], tomo_stage.y, [tomo_stage.y.position-wf_sam_movry]) 
    
    #collecting tomography data
    print('start collecting tomographyd data')
    print('start anggle', thetastart)
    print('stop anggle', thetastop)
    print('number of projections', numproj)    
    
    fftomo_plan = scan(det, tomo_stage.theta, thetastart, thetastop, numproj)        
    fftomo_plan = bp.subs_wrapper(fftomo_plan, livecallbacks)
    fftomo_gen = yield from fftomo_plan
    logscan_event0info('full_field_tomo_projections', event0info = eventlog_list)
    #logscan('full_field_tomo_projections')          
    
    #move sample out prior to white field collection
    print('moving sample out of field of view')
    print('moving sample x relative by', wf_sam_movrx)
    print('moving sample y relative by', wf_sam_movry)
    movesamplex_out = yield from list_scan([], tomo_stage.x, [tomo_stage.x.position+wf_sam_movrx])
    movesampley_out = yield from list_scan([], tomo_stage.y, [tomo_stage.y.position+wf_sam_movry])
    
    time.sleep(wfwait)
    #collecting whitefield    
    fftomo_wf_plan = count(det, num = num_whitefield)
    fftomo_wf_plan = bp.subs_wrapper(fftomo_wf_plan, livecallbacks)
    fftomo_wf_gen = yield from fftomo_wf_plan
    logscan_event0info('full_field_tomo_white_field_postscan', event0info = eventlog_list)
    #logscan('full_field_tomo_white_field_postscan')          

    #move sample back after white field collection
    print('moving sample back to the field of view')
    print('moving sample x relative by', (-1)*wf_sam_movrx)
    print('moving sample y relative by', (-1)*wf_sam_movry)
    movesamplex_in = yield from list_scan([], tomo_stage.x, [tomo_stage.x.position-wf_sam_movrx])
    movesampley_in = yield from list_scan([], tomo_stage.y, [tomo_stage.y.position-wf_sam_movry])    
    
    print('done')
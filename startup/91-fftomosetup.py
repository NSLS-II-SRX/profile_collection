# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:13 2016

@author: xf05id1
"""

#for tomography
from bluesky.plans import subs_wrapper, scan, count, list_scan
  
def tomo_fullfield(thetastart = -90, thetastop = 90, numproj = 361,
            acqtime = 0.002, 
            num_darkfield = 10, 
            num_whitefiled = 10, wf_sam_movrx = 0, wf_sam_movry = -2,
            ):
                
    '''
    num_whitefield (int, default = 10): number of white field images collected before and after the scan
    wf_sam_movrx (float, default = 0): distance to move x before/after the tomography collection to collect white field
    wf_sam_movry (float, default = -2): distance to move y before/after the tomography collection to collect white field
    '''               
    theta_traj = np.linspace(thetastart, thetastop, numproj)
    tomo_scan_output = []
    
    livecallbacks = []    
    livetableitem = ['tomo_stage_theta', 'current_preamp_ch2']
    livecallbacks.append(LiveTable(livetableitem))
        
    #setup detectors
    det = [current_preamp, pcoedge, tomo_stage.theta]
    current_preamp.exp_time.put(acqtime)
    xs.settings.acquire_time.put(acqtime)

    #close the shutter
    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)    
    #collecting darkfield
    fftomo_df_plan = count(det, num = num_darkfield)
    fftomo_df_plan = bp.subs_wrapper(fftomo_df_plan, livecallbacks)
    fftomo_df_gen = yield from fftomo_df_plan
    logscan('fule_field_tomo_dark_field') 

    #open the shutter
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1)      

    #move sample out prior to white field collection
    movesamplex_out = yield from relative_list_scan([], tomo_stage.x, [wf_sam_movrx])
    movesampley_out = yield from relative_list_scan([], tomo_stage.y, [wf_sam_movry])

    #collecting whitefield        
    fftomo_wf_plan = count(det, num = num_whitefield)
    fftomo_wf_plan = bp.subs_wrapper(fftomo_wf_plan, livecallbacks)
    fftomo_wf_gen = yield from fftomo_wf_plan
    logscan('fule_field_tomo_white_field_prescan')      

    #move sample back after white field collection
    movesamplex_in = yield from relative_list_scan([], tomo_stage.x, [(-1)*wf_sam_movrx])
    movesampley_in = yield from relative_list_scan([], tomo_stage.y, [(-1)*wf_sam_movry])    
    
    fftomo_plan = scan(det, tomo_stage.theta, thetastart, thetastop, numproj)        
    fftomo_plan = bp.subs_wrapper(fftomo_plan, livecallbacks)
    fftomo_gen = yield from fftomo_plan
    logscan('fule_field_tomo_projections')          
    
    #move sample out prior to white field collection
    movesamplex_out = yield from relative_list_scan([], tomo_stage.x, [wf_sam_movrx])
    movesampley_out = yield from relative_list_scan([], tomo_stage.y, [wf_sam_movry])

    #collecting whitefield    
    fftomo_wf_plan = count(det, num = num_whitefield)
    fftomo_wf_plan = bp.subs_wrapper(fftomo_wf_plan, livecallbacks)
    fftomo_wf_gen = yield from fftomo_wf_plan
    logscan('fule_field_tomo_white_field_postscan')          

    #move sample back after white field collection
    movesamplex_in = yield from relative_list_scan([], tomo_stage.x, [(-1)*wf_sam_movrx])
    movesampley_in = yield from relative_list_scan([], tomo_stage.y, [(-1)*wf_sam_movry])    
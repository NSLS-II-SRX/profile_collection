print(f'Loading {__file__}...')

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:13 2016

@author: xf05id1
"""

#for tomography
from bluesky.plans import scan, count, list_scan
from bluesky.preprocessors import subs_wrapper
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
    shut_b.close()
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
    shut_b.open()

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
print(f'Loading {__file__}...')

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:40:40 2016

@author: xf05id1
"""
from bluesky.plans import scan, count, list_scan
from bluesky.preprocessors import subs_wrapper
import numpy as np

def xanes_afterscan_pco(scanid, roinum, filename, i0scale, itscale, roi_key):
    
    #roinum=0, i0scale = 1e8, itscale = 1e8, 
    #xanes_afterscan(789, 0, '118lw_palmiticontopetoh85rh', 1e8, 1e8)

    print(scanid)
    headeritem = [] 
    h=db[scanid]

    #datatable = get_table(h, ['energy_energy', 'saturn_mca_rois_roi'+str(roinum)+'_net_count', 'current_preamp_ch2'],
    #                      stream_name='primary')
    #
    #energy_array = list(datatable['energy_energy'])   
    #if_array = list(datatable['saturn_mca_rois_roi'+str(roinum)+'_net_count'])
    #i0_array = list(datatable['current_preamp_ch2'])
    
    userheaderitem = {}
    userheaderitem['sample.name'] = h.start['sample']['name']
    userheaderitem['initial_sample_position.hf_stage.x'] = h.start['initial_sample_position']['hf_stage_x']
    userheaderitem['initial_sample_position.hf_stage.y'] = h.start['initial_sample_position']['hf_stage_y']
    
    userheaderitem['wb_slits.h_cen'] = h.start['wb_slits']['h_cen']
    userheaderitem['wb_slits.h_gap'] = h.start['wb_slits']['h_gap']
    userheaderitem['wb_slits.v_cen'] = h.start['wb_slits']['v_cen']
    userheaderitem['wb_slits.v_gap'] = h.start['wb_slits']['v_gap']

    userheaderitem['hfm.y'] = h.start['hfm']['y']
    userheaderitem['hfm.bend'] = h.start['hfm']['bend']

    userheaderitem['ssa_slits.h_cen'] = h.start['ssa_slits']['h_cen']
    userheaderitem['ssa_slits.h_gap'] = h.start['ssa_slits']['h_gap']
    userheaderitem['ssa_slits.v_cen'] = h.start['ssa_slits']['v_cen']
    userheaderitem['ssa_slits.v_gap'] = h.start['ssa_slits']['v_gap']    
    
    
    print(userheaderitem['initial_sample_position.hf_stage.x'])
    print(userheaderitem['initial_sample_position.hf_stage.y'])
        
    #columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_net_count','saturn_mca_rois_roi'+str(roinum)+'_count', 'current_preamp_ch0', 'current_preamp_ch2']    
    columnitem = ['energy_energy', 'energy_u_gap_readback', 'energy_bragg', 'energy_c2_x',
                  'current_preamp_ch0', 'current_preamp_ch2', roi_key[0], roi_key[1], roi_key[2],
                   'pcoedge_stats1_total', 'pcoedge_stats2_total', 'pcoedge_stats3_total', 'pcoedge_stats4_total']    
    
    usercolumnitem = {}
 
    #usercolumnnameitem = ['scaled_current_preamp_ch0', 'scaled_current_preamp_ch2', 'roi_sum']
    usercolumnnameitem = ['I0', 'It', 'If']
    
    datatable = get_table(h, ['current_preamp_ch0', 'current_preamp_ch2', roi_key[0], roi_key[1], roi_key[2]],
                          stream_name='primary')        
    i0_array = abs(numpy.array(datatable['current_preamp_ch2']) - i0_baseline) * i0scale
    it_array = abs(numpy.array(datatable['current_preamp_ch0']) - it_baseline) * itscale
    roi_sum = numpy.array(datatable[roi_key[0]]) +  numpy.array(datatable[roi_key[1]]) + numpy.array(datatable[roi_key[2]])  
   
    usercolumnitem['I0'] = i0_array
    usercolumnitem['It'] = it_array
    usercolumnitem['If'] = roi_sum
    
    
    scanoutput.textout(scan = scanid, header = headeritem, userheader = userheaderitem, column = columnitem, 
                       usercolumn = usercolumnitem, usercolumnname = usercolumnnameitem, output = False, filename_add = filename) 

def pco_zscan(acqtime = 0.0005, record_preamp = True, preamp_acqtime = None,
            num_img = 10, imgwait = 2, 
            pcoedge_zstart = 9, pcoedge_zstop = 49, num_step = 11,
            eventlog_list = ['pcoedge_tiff_file_name', 'pcoedge_cam_acquire_time']
            ):
    
    print('acqusition time', acqtime)
 
    
    #setup detectors
    det = [pcoedge]
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
    livetableitem = ['pcoedge_stats1_total', 'pcoedge_stats2_total', 'pcoedge_stats3_total', 'pcoedge_stats4_total']    
    if record_preamp is True:
        livetableitem.append('current_preamp_ch2')
    livecallbacks.append(LiveTable(livetableitem))
    
    for pcoedg_pos_ztar in np.linspace(pcoedge_zstart, pcoedge_zstop, num_step): 
        #move detecotor z
        #movesamplex_out = yield from list_scan([], pcoedge_pos.z, [pcoedg_pos_ztar])
        print('moving detector z')
        mov(pcoedge_pos.z, pcoedg_pos_ztar)
              
        #collecting images
        time.sleep(imgwait)
        imgplan = count(det, num = num_img)
        imgplan = bp.subs_wrapper(imgplan, livecallbacks)
        imgplan_gen = yield from imgplan
        logscan_event0info('pcoedge_image', event0info = eventlog_list)
        #logscan('full_field_tomo_dark_field')     
        print('collection done at pcoedge z', pcoedg_pos_ztar)
    
    print('done')
    
def pco_xanes(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True, correct_c1_r = False,             
            acqtime=None, roinum=1, i0scale = 1e8, itscale = 1e8, delaytime = 0.2,
            samplename = '', filename = '', 
            eventlog_list = ['pcoedge_tiff_file_name', 'pcoedge_cam_acquire_time']):
                
    '''

    '''                                
                
    #make sure user provided correct input

    if erange is []:
        raise Exception('erange = [], must specify energy ranges')
    if estep is []:
        raise Exception('estep = [], must specify energy step sizes')
    if len(erange)-len(estep) is not 1:
        raise Exception('must specify erange and estep correctly.'\
                         +'e.g. erange = [7000, 7100, 7150, 7500], estep = [2, 0.5, 5] ')

    if acqtime is None:
        raise Exception('acqtime = None, must specify an acqtime position')

    #record relevant meta data in the Start document, defined in 90-usersetup.py
    metadata_record()
    
    #convert erange and estep to numpy array
    erange = numpy.array(erange)
    estep = numpy.array(estep)

    #calculation for the energy points        
    ept = numpy.array([])
    for i in range(len(estep)):
        ept = numpy.append(ept, numpy.arange(erange[i], erange[i+1], estep[i]))
    ept = numpy.append(ept, numpy.array(erange[-1]))
    ept = ept/1000
   
    
    #setup the detector
    current_preamp.exp_time.put(acqtime-delaytime)
    xs.settings.acquire_time.put(acqtime)
    xs.total_points.put(len(ept))
   
    det = [current_preamp, xs, ring_current, pcoedge]

    #setup the live callbacks
    livecallbacks = []    
    
    livetableitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch2', 'pcoedge_stats1_total', 'pcoedge_stats2_total', 'pcoedge_stats3_total']  
    
    roi_name = 'roi{:02}'.format(roinum)
    
    roi_key = []
    roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
    roi_key.append(getattr(xs.channel2.rois, roi_name).value.name)
    roi_key.append(getattr(xs.channel3.rois, roi_name).value.name)
    livetableitem.append(roi_key[0])    

    livecallbacks.append(LiveTable(livetableitem))

    liveploty = roi_key[0]
    liveplotx = energy.energy.name
    liveplotfig = plt.figure('raw xanes')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))

    liveploty = 'current_preamp_ch2'
    liveplotfig2 = plt.figure('i0')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    
    livenormfig = plt.figure('normalized xanes')    
    i0 = 'current_preamp_ch2'
    livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, fig=livenormfig))  

    liveploty = 'pcoedge_stats1_total'
    liveplotfig3 = plt.figure('pcoedge_stats1_total')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig3))

    liveploty = 'pcoedge_stats2_total'
    liveplotfig4 = plt.figure('pcoedge_stats2_total')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig4))
    
    liveploty = 'pcoedge_stats3_total'
    liveplotfig5 = plt.figure('pcoedge_stats3_total')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig5))

    if correct_c2_x is False:
        energy.move_c2_x.put(False)
    elif correct_c2_x is True:
    #this line sets the current xoffset to the energy axis, might want to see if that's what we want to do in the long run
        energy._xoffset = energy.crystal_gap()
    else:
    #user can set correct_c2x to a float value and force the energy._xoffset to that value, useful when changing between edges in a batch
        energy._xoffset = correct_c2_x
        
    if correct_c1_r is not False:
        dcm.c1_roll.set(correct_c1_r)
        
    if harmonic is not None:        
        energy.harmonic.put(harmonic)

    #add user meta data
    gs.RE.md['sample']  = {'name': samplename}
    #setup the plan
    ept = list(ept)
    xanes_scanplan = AbsListScanPlan(det, energy, ept)
    ept = numpy.array(ept)

    #run the plan
    scaninfo = gs.RE(xanes_scanplan, livecallbacks, raise_if_interrupted=True)

    print(type(scaninfo))
    print(scaninfo)

    #output the datafile
    xanes_afterscan_pco(scaninfo[0], roinum, filename, i0scale, itscale, roi_key)

    logscan_event0info('pco_xanes', event0info = eventlog_list) 

    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)
                  
    return scaninfo[0]

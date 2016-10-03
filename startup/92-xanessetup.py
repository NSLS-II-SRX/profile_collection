# -*- coding: utf-8 -*-
"""
set up for XANES scan for HF mode at SRX

Created on Fri Feb 19 12:16:52 2016
Modified on Wed Wed 02 14:14 to comment out the saturn detector which is not in use

@author: xf05id1
"""

from bluesky.plans import AbsListScanPlan
import scanoutput
import numpy
import time
from epics import PV

i0_baseline = 7.24e-10
it_baseline = 1.00e-7

dcm_bragg_temp_pv = 'XF:05IDA-OP:1{Mono:HDCM-Ax:P}T-I'
dcm_bragg_temp_pv_epics = PV(dcm_bragg_temp_pv)
bragg_waittime = 30*60

def xanes_afterscan(scanid, roinum, filename, i0scale, itscale, roi_key):
    
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
                  'current_preamp_ch0', 'current_preamp_ch2', roi_key[0], roi_key[1], roi_key[2]]    
    
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
    
    #ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))


def xanes_afterscan_tmode(scanid, filename, i0scale, itscale):
    
    #roinum=0, i0scale = 1e8, itscale = 1e8, 
    #xanes_afterscan(789, 0, '118lw_palmiticontopetoh85rh', 1e8, 1e8)

    print(scanid)
    headeritem = [] 
    h=db[scanid]
    
    userheaderitem = {}
    userheaderitem['sample.name'] = h.start['sample']['name']
    userheaderitem['initial_sample_position.hf_stage.x'] = h.start['initial_sample_position']['hf_stage_x']
    userheaderitem['initial_sample_position.hf_stage.y'] = h.start['initial_sample_position']['hf_stage_y']



    
    print(userheaderitem['initial_sample_position.hf_stage.x'])
    print(userheaderitem['initial_sample_position.hf_stage.y'])
    

    columnitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch2']    
    
    usercolumnitem = {}
 
    #usercolumnnameitem = ['scaled_current_preamp_ch0', 'scaled_current_preamp_ch2', 'roi_sum']
    usercolumnnameitem = ['I0', 'It']
    
    datatable = get_table(h, ['current_preamp_ch0', 'current_preamp_ch2'], stream_name='primary')        
    i0_array = abs(numpy.array(datatable['current_preamp_ch2']) - i0_baseline) * i0scale
    it_array = abs(numpy.array(datatable['current_preamp_ch0']) - it_baseline) * itscale
   
    usercolumnitem['I0'] = i0_array
    usercolumnitem['It'] = it_array
   
    scanoutput.textout(scan = scanid, header = headeritem, userheader = userheaderitem, column = columnitem, 
                       usercolumn = usercolumnitem, usercolumnname = usercolumnnameitem, output = False, filename_add = filename) 

def xanes(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True, correct_c1_r = False,             
            acqtime=None, roinum=1, i0scale = 1e8, itscale = 1e8, delaytime = 0.2,
            samplename = '', filename = ''):
                
    '''
    erange (list of float): energy ranges for XANES in eV, e.g. erange = [7112-50, 7112-20, 7112+50, 7112+120]
    estep  (list of float): energy step size for each energy range in eV, e.g. estep = [2, 1, 5]
    
    harmonic (None or odd integer): when set to None, use the highest harmonic achievable automatically. 
                                    when set to an odd integer, force the XANES scan to use that harmonic
                                    this is important for energy range that might cross harmonic.
    correct_c2_x (boolean or float): when True, automatically correct the c2x. 
                                                 Note that it will set energy._xoffset = energy.crystal_gap(), which is the crystal gap based on the current c2x value.                                    
                                     when False, c2x will not be moved during the XANES scan
                                     when set to a float number, it will set the energy._xoffset to this value. See 98-cunysetup.py changeE function for example.
    correct_c1_r (False or float): when False, c1r will not be moved during a XANES scan
                                   when set to a float, c1r will be set to that value before a XANES scan but will remain the same during the whole scan.

    acqtime (float): acqusition time to be set for both xspress3 and F460                                   
    roinum: setup the roi to be used to calculate the XANES spectrum. 
    
    i0scale (float): default to 1e8, all i0 values will be scaled by this number
    itscale (float): default to 1e8, all it values will be scaled by this number
    
    samplename (string): sample name to be saved in metadata
    filename (string): filename to be attached to the scan id as the text output file.

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
    #saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    #saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.net_count')       
    #saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.count') 
    #det = [current_preamp, saturn, ring_current]        
    #det = [current_preamp, ring_current]        
    det = [current_preamp, xs, ring_current]

    #setup the live callbacks
    livecallbacks = []    
    
    livetableitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch2']  
    
    roi_name = 'roi{:02}'.format(roinum)
    
    roi_key = []
    roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
    roi_key.append(getattr(xs.channel2.rois, roi_name).value.name)
    roi_key.append(getattr(xs.channel3.rois, roi_name).value.name)
    livetableitem.append(roi_key[0])    
    
    #livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_net_count')
    #livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_count')
    livecallbacks.append(LiveTable(livetableitem))

    #liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_net_count'
    #liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_count'
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
    ept = numpy.array(ept, dtype=float)

    #open b shutter
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1) 

    #run the plan
    scaninfo = gs.RE(xanes_scanplan, livecallbacks, raise_if_interrupted=True)

    #close b shutter
    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)

    print(type(scaninfo))
    print(scaninfo)

    #output the datafile
    xanes_afterscan(scaninfo[0], roinum, filename, i0scale, itscale, roi_key)

    logscan('xanes') 

    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)
                  
    return scaninfo[0]
    


def xanes_tmode(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True, correct_c1_r = False,             
            acqtime=None, i0scale = 1e8, itscale = 1e8,
            samplename = '', filename = ''):
                
    '''
    this function doesn't use fluorecence detector but use only the transmission diode

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
    current_preamp.exp_time.put(acqtime)     
    det = [current_preamp, ring_current]

    #setup the live callbacks
    livecallbacks = []    
    
    livetableitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch2']  
    livecallbacks.append(LiveTable(livetableitem))

    liveplotx = energy.energy.name
    liveploty = 'current_preamp_ch0'
    liveplotfig = plt.figure('it')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))

    liveploty = 'current_preamp_ch2'
    liveplotfig2 = plt.figure('i0')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    
#    livenormfig = plt.figure('normalized xanes')    
#    i0 = 'current_preamp_ch2'
#    livecallbacks.append(NormalizeLivePlot(roi_key[0], x=liveplotx, norm_key = i0, fig=livenormfig))  

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
    ept = [float(_) for _ in ept]
    xanes_scanplan = AbsListScanPlan(det, energy, ept)
    ept = numpy.array(ept)

   #open b shutter
    shut_b.open_cmd.put(1)
    while (shut_b.close_status.get() == 1):
        epics.poll(.5)
        shut_b.open_cmd.put(1) 

    #run the plan
    scaninfo = gs.RE(xanes_scanplan, livecallbacks, raise_if_interrupted=True)

    #close b shutter
    shut_b.close_cmd.put(1)
    while (shut_b.close_status.get() == 0):
        epics.poll(.5)
        shut_b.close_cmd.put(1)

    print(type(scaninfo))
    print(scaninfo)

    #output the datafile
    #xanes_afterscan(scaninfo[0], roinum, filename, i0scale, itscale, roi_key)
    xanes_afterscan_tmode(scaninfo[0], filename, i0scale, itscale)

    logscan('xanes') 

    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)
                  
    return scaninfo[0]
  
    
def hfxanes_xybatch(xylist=[], waittime = None, 
                    samplename = None, filename = None,
                    erange = [], estep = [],  
                    harmonic = None, correct_c2_x=True, delaytime=0.2,             
                    acqtime=None, roinum=1, i0scale = 1e8, itscale = 1e8,
                    ):
                        
    '''
    Running batch XANES scans on different locations, defined as in xylist.
    input: 
        xylist (list of x,y positions in float): pairs of x, y positions on which XANES scans will be collected
            E.g. xylist = [[10.4, 20.4], [10.5, 20.8]] 
        waitime (list of float): wait time between scans, if not specified, 2 seconds will be used
            E.g. waittime = [10] #10 sec. wait time will be used between all scans
            E.g. waititme = [10, 20] #10 sec. will be used between 1st and 2nd scans; 20 sec. will be used after the 2nd scan. The number of scans need to match with the number of waittime listed
        samplename (list of string): list of sample names to be used.
            If with one component, all scans will be set to the same sample name
            If with more than one component, the lenth of the list must match the lenth of the xylist. The sample name will then be assigned 1-1.
            E.g. samplename = ['sample1']: all scans will have the same sample name
            E.g. samplename = ['sample1', 'sample2']: two points in the xylist will have different sample names
        filename (list of string): list of file names to be used
            same rules as in sample name is used.
            E.g. filename = ['sample1']: all scans will have the same file name
            E.g. filename = ['sample1', 'sample2']: two points in the xylist will have different file names attached to their scan ids.
                       
        other inputs are same as in the xanes funciton.
    '''
    
    for pt_num, position in enumerate(xylist):
        #move stages to the next point
        hf_stage.x.set(position[0]) 
        hf_stage.y.set(position[1])

        #check bragg temperature before start the scan
        if dcm_bragg_temp_pv_epics.get() > 110:
            print('bragg temperature too high, wait ' + str(bragg_waittime) + ' s.')            
            time.sleep(bragg_waittime)
        
        time.sleep(3)
        
        
        print(len(samplename))        
        
        if samplename is None:
            pt_samplename = ''
        else:
            if len(samplename) is 1:
                pt_samplename = samplename[0]                
            elif len(samplename) is not len(xylist):
                err_msg = 'number of samplename is different from the number of points'
                raise Exception(err_msg)            
            else:
                pt_samplename = samplename[pt_num]

        if filename is None:
            pt_filename = ''
        else:
            if len(filename) is 1:
                pt_filename = filename[0]     
            elif len(filename) is not len(xylist):
                err_msg = 'number of filename is different from the number of points'
                raise Exception(err_msg)
            else:
                pt_filename = filename[pt_num]
                
        
        xanes(erange = erange, estep = estep,  
            harmonic = harmonic, correct_c2_x= correct_c2_x,              
            acqtime = acqtime, roinum = roinum, 
            i0scale = i0scale, itscale = itscale, delaytime=delaytime,
            samplename = pt_samplename, filename = pt_filename)
            
                #wait for specified time period in sec.
        if waittime is None:
            time.sleep(2)
        elif len(waittime) is 1:
            time.sleep(waittime[0])
        elif len(samplename) is not len(waittime):
            err_msg = 'number of waittime is different from the number of points'
        else:
            time.sleep(waittime[pt_num])

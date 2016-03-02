# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:16:52 2016

set up for 2D XRF scan for HR mode
@author: xf05id1
"""


from bluesky.plans import AbsListScanPlan
from bluesky.suspenders import PVSuspendFloor
import scanoutput
import numpy
import time

ring_current_pv = 'SR:C03-BI{DCCT:1}I:Real-I'
cryo_v19_pv = 'XF:05IDA-UT{Cryo:1-IV:19}Sts-Sts'
i0_baseline = 7.24e-10
it_baseline = 1.40e-8

def xanes_afterscan(scanid, roinum, filename, i0scale, itscale):
    
    #roinum=0, i0scale = 1e8, itscale = 1e8, 
    #xanes_afterscan(789, 0, '118lw_palmiticontopetoh85rh', 1e8, 1e8)

    print(scanid)
    headeritem = [] 
    h=db[scanid]

    #datatable = get_table(h, ['energy_energy', 'saturn_mca_rois_roi'+str(roinum)+'_net_count', 'current_preamp_ch2'])        
    #energy_array = list(datatable['energy_energy'])   
    #if_array = list(datatable['saturn_mca_rois_roi'+str(roinum)+'_net_count'])
    #i0_array = list(datatable['current_preamp_ch2'])
    
    userheaderitem = {}
    userheaderitem['sample.name'] = h.start['sample']['name']
    userheaderitem['initial_sample_position.hf_stage.x'] = h.start['initial_sample_position']['hf_stage_x']
    userheaderitem['initial_sample_position.hf_stage.y'] = h.start['initial_sample_position']['hf_stage_y']
    
    print(userheaderitem['initial_sample_position.hf_stage.x'])
    print(userheaderitem['initial_sample_position.hf_stage.y'])
    

    
    #columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_net_count', 'current_preamp_ch2']
    columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_net_count','saturn_mca_rois_roi'+str(roinum)+'_count', 'current_preamp_ch0', 'current_preamp_ch2']    
    #columnitem = ['energy_energy','saturn_mca_rois_roi'+str(roinum)+'_count', 'current_preamp_ch2']    
    
    usercolumnitem = {}
 
    usercolumnnameitem = ['scaled_current_preamp_ch0', 'scaled_current_preamp_ch2']
    datatable = get_table(h, ['current_preamp_ch0', 'current_preamp_ch2'])        
    i0_array = abs(numpy.array(datatable['current_preamp_ch2']) - i0_baseline) * i0scale
    it_array = abs(numpy.array(datatable['current_preamp_ch0']) - it_baseline) * itscale
   
    usercolumnitem['scaled_current_preamp_ch2'] = i0_array
    usercolumnitem['scaled_current_preamp_ch0'] = it_array
    
    scanoutput.textout(scan = scanid, header = headeritem, userheader = userheaderitem, column = columnitem, 
                       usercolumn = usercolumnitem, usercolumnname = usercolumnnameitem, output = False, filename_add = filename) 
    
    #ps.append(PeakStats(energy.energy.name, bpmAD.stats3.total.name))

def xanes(erange = [], estep = [],  
            harmonic = None, correct_c2_x=True,              
            acqtime=None, roinum=0, i0scale = 1e8, itscale = 1e8,
            samplename = '', filename = ''):
                
    '''
    roinum: setup the roi to be used to calculate the XANES spectrum

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
    saturn.mca.preset_real_time.put(acqtime)
    #saturn.mca.preset_live_time.put(acqtime)

    saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.net_count')       
    saturn.read_attrs.append('mca.rois.roi'+str(roinum)+'.count') 
    det = [current_preamp, saturn, ring_current]        

    #setup the live callbacks
    livecallbacks = []    
    
    livetableitem = ['energy_energy', 'current_preamp_ch0', 'current_preamp_ch2']    
    livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_net_count')
    livetableitem.append('saturn_mca_rois_roi'+str(roinum)+'_count')
    livecallbacks.append(LiveTable(livetableitem, max_post_decimal = 4))

    #liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_net_count'
    liveploty = 'saturn_mca_rois_roi'+str(roinum)+'_count'
    liveplotx = energy.energy.name
    liveplotfig = plt.figure()
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))

    liveploty = 'current_preamp_ch2'
    liveplotfig2 = plt.figure()
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig2))
    
    livenormfig = plt.figure()    
    i0 = 'current_preamp_ch2'
    livecallbacks.append(NormalizeLivePlot(liveploty, x=liveplotx, norm_key = i0, fig=livenormfig))  
    livenormfig2 = plt.figure()    
    livecallbacks.append(NormalizeLivePlot('current_preamp_ch0', x=liveplotx, norm_key = i0, fig=livenormfig2))  



    if correct_c2_x is False:
        energy.move_c2_x.put(False)
    else:
    #this line sets the current xoffset to the energy axis, might want to see if that's what we want to do in the long run
        energy._xoffset = energy.crystal_gap()
        
    if harmonic is not None:        
        energy.harmonic.put(harmonic)

    #add user meta data
    gs.RE.md['sample']  = {'name': samplename}

    #setup the plan  
    xanes_scanplan = AbsListScanPlan(det, energy, ept)

    #run the plan
    scaninfo = gs.RE(xanes_scanplan, livecallbacks, raise_if_interrupted=True)

    #output the datafile
    xanes_afterscan(scaninfo, roinum, filename, i0scale, itscale)

    logscan('xanes') 

    #clean up when the scan is done    
    energy.move_c2_x.put(True)
    energy.harmonic.put(None)
    
              
    return scaninfo
    
def hfxanes_xybatch(xylist=[], waittime = 5, 
                    samplename = None, filename = None,
                    erange = [], estep = [],  
                    harmonic = None, correct_c2_x=True,              
                    acqtime=None, roinum=0, i0scale = 1e8, itscale = 1e8,
                    ):
    for pt_num, position in enumerate(xylist):
        #move stages to the next point
        hf_stage.x.set(position[0]) 
        hf_stage.y.set(position[1])
        
        #wait for specified time period in sec.
        time.sleep(waittime)
        
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
            i0scale = i0scale, itscale = itscale,
            samplename = pt_samplename, filename = pt_filename)
